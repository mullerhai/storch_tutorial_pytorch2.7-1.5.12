
import scala.collection.mutable
import scala.util.Using
import torch.*
import torch.nn.*
import torch.optim.*
import torch.utils.data.*

// 下载并解压数据集
def downloadAndExtract(url: String, savePath: String): Unit =
  if!Files.exists(Paths.get(savePath)) then
    Files.createDirectories(Paths.get(savePath))
  val zipPath = Paths.get(savePath, "creditcard.zip").toString
  if!Files.exists(Paths.get(zipPath)) then
    println("Downloading dataset...")
    val in = new URL(url).openStream()
    Files.copy(in, Paths.get(zipPath))
    in.close()
  println("Extracting dataset...")
  val zipFile = new java.util.zip.ZipFile(zipPath)
  val entries = zipFile.entries()
  while entries.hasMoreElements() do
    val entry = entries.nextElement()
    val entryPath = Paths.get(savePath, entry.getName)
    if entry.isDirectory() then
      Files.createDirectories(entryPath)
    else
      Files.copy(zipFile.getInputStream(entry), entryPath)
  zipFile.close()

// 自定义数据集类
class CreditFraudDataset(features: Tensor[Float32], labels: Tensor[Float32]) extends Dataset[Tensor[Float32], Tensor[Float32]]:
  def len(): Int = labels.size(0)
  def apply(idx: Int): (Tensor[Float32], Tensor[Float32]) = (features(idx), labels(idx))

// 专家网络
class Expert(inputDim: Int, hiddenDim: Int) extends Module:
  val fc1 = new Linear(inputDim, hiddenDim)
  val fc2 = new Linear(hiddenDim, 1)
  val relu = new ReLU()

  def forward(x: Tensor[Float32]): Tensor[Float32] =
    val out = relu(fc1(x))
    fc2(out)

// 门控网络
class GatingNetwork(inputDim: Int, numExperts: Int) extends Module:
  val fc = new Linear(inputDim, numExperts)
  val softmax = new Softmax(dim = 1)

  def forward(x: Tensor[Float32]): Tensor[Float32] =
    softmax(fc(x))

// 混合专家模型
class MoE(inputDim: Int, hiddenDim: Int, numExperts: Int) extends Module:
  val experts = ModuleList[Expert](Seq.fill(numExperts)(new Expert(inputDim, hiddenDim)))
  val gatingNetwork = new GatingNetwork(inputDim, numExperts)

  def forward(x: Tensor[Float32]): Tensor[Float32] =
    val gates = gatingNetwork(x)
    val expertOutputs = experts.map(_(x))
    val stackedOutputs = torch.stack(expertOutputs, dim = 1).squeeze(-1)
    torch.sum(gates.unsqueeze(-1) * stackedOutputs, dim = 1)

// 评分卡模块
class ScoreCard(inputDim: Int, A: Double = 600, B: Double = 50 / np.log(2)) extends Module:
  val logisticRegression = new Linear(inputDim, 1)
  val sigmoid = new Sigmoid()

  def forward(x: Tensor[Float32]): Tensor[Float32] =
    val logits = logisticRegression(x)
    val prob = sigmoid(logits)
    val odds = prob / (1 - prob)
    val score = A - B * torch.log(odds)
    score

// Transformer MoE 模型
class TransformerMoE(inputDim: Int, hiddenDim: Int, numExperts: Int, nhead: Int = 4, numLayers: Int = 2) extends Module:
  val moe = new MoE(inputDim, hiddenDim, numExperts)
  val scoreCard = new ScoreCard(inputDim)
  val transformerEncoder = new TransformerEncoder(
    new TransformerEncoderLayer(dModel = inputDim, nhead = nhead),
    numLayers = numLayers
  )
  val fc = new Linear(inputDim, 1)
  val sigmoid = new Sigmoid()

  def forward(x: Tensor[Float32]): Tensor[Float32] =
    val xTransformer = transformerEncoder(x.unsqueeze(0)).squeeze(0)
    val moeOutput = moe(xTransformer)
    val scoreCardOutput = scoreCard(xTransformer)
    val combinedOutput = moeOutput + scoreCardOutput
    val output = fc(combinedOutput)
    sigmoid(output)

// 训练函数
def train(model: Module, trainLoader: DataLoader[Tensor[Float32], Tensor[Float32]], criterion: Loss[Float32], optimizer: Optimizer, device: Device): Double =
  model.train()
  var totalLoss = 0.0
  for (features, labels) <- trainLoader do
    val featuresDevice = features.to(device)
    val labelsDevice = labels.to(device)
    optimizer.zeroGrad()
    val outputs = model(featuresDevice).asInstanceOf[Tensor[Float32]].squeeze()
    val loss = criterion(outputs, labelsDevice)
    loss.backward()
    optimizer.step()
    totalLoss += loss.item()
  totalLoss / trainLoader.length

// 评估函数
def evaluate(model: Module, testLoader: DataLoader[Tensor[Float32], Tensor[Float32]], criterion: Loss[Float32], device: Device): Double =
  model.eval()
  var totalLoss = 0.0
  torch.no_grad {
    for (features, labels) <- testLoader do
      val featuresDevice = features.to(device)
      val labelsDevice = labels.to(device)
      val outputs = model(featuresDevice).asInstanceOf[Tensor[Float32]].squeeze()
      val loss = criterion(outputs, labelsDevice)
      totalLoss += loss.item()
  }
  totalLoss / testLoader.length

// 主函数
@main def main(): Unit =
  // 下载并解压数据集
  val url = "https://storage.googleapis.com/download.tensorflow.org/data/creditcard.zip"
  val savePath = "data"
  downloadAndExtract(url, savePath)

  // 读取数据
  val dataPath = Paths.get(savePath, "creditcard.csv").toString
  val df = DataFrame.read_csv(dataPath)

  // 数据预处理
  val X = df.drop("Class").values.toTensor[Float32]
  val y = df("Class").values.toTensor[Float32]

  val scaler = new StandardScaler()
  val XScaled = scaler.fit_transform(X)

  val Array(XTrain, XTest, yTrain, yTest) = train_test_split(XScaled, y, testSize = 0.2, randomState = 42)

  val trainDataset = new CreditFraudDataset(XTrain, yTrain)
  val testDataset = new CreditFraudDataset(XTest, yTest)

  val trainLoader = DataLoader(trainDataset, batchSize = 64, shuffle = true)
  val testLoader = DataLoader(testDataset, batchSize = 64, shuffle = false)

  // 检查 GPU 是否可用
  val device = if torch.cuda.isAvailable() then torch.device("cuda") else torch.device("cpu")

  // 初始化模型
  val inputDim = XTrain.size(1)
  val hiddenDim = 64
  val numExperts = 4
  val model = new TransformerMoE(inputDim, hiddenDim, numExperts).to(device)

  // 定义损失函数和优化器
  val criterion = new BCELoss()
  val optimizer = new Adam(model.parameters(), lr = 0.001)

  // 训练模型
  val numEpochs = 10
  for epoch <- 1 to numEpochs do
    val trainLoss = train(model, trainLoader, criterion, optimizer, device)
    val testLoss = evaluate(model, testLoader, criterion, device)
    println(f"Epoch $epoch/$numEpochs, Train Loss: $trainLoss%.4f, Test Loss: $testLoss%.4f")
