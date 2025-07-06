
import scala.collection.mutable.ArrayBuffer
import torch.*
import torch.nn.*
import torch.optim.*
import torch.utils.data.*

// 检查 GPU 是否可用
val device = if torch.cuda.isAvailable() then torch.device("cuda") else torch.device("cpu")
println(s"Using device: $device")

// 数据路径
val DATA_DIR = "./data/avazu"
val TRAIN_PATH = new File(DATA_DIR, "train.gz").getPath
val TEST_PATH = new File(DATA_DIR, "test.gz").getPath

// 检查文件是否存在
val filePaths = Seq(TRAIN_PATH, TEST_PATH)
if !filePaths.forall(path => new File(path).exists()) then
  throw new java.io.FileNotFoundException("请检查 Avazu 数据集文件是否存在于指定目录。")

// 读取并解压数据
def readGzFile(filePath: String): DataFrame =
  val gzStream = new GZIPInputStream(new java.io.FileInputStream(filePath))
  val df = DataFrame.read_csv(gzStream)
  gzStream.close()
  df

val trainDf = readGzFile(TRAIN_PATH)
val testDf = readGzFile(TEST_PATH)
println("读取并解压数据完成。")

// 数据预处理
// 训练集的 'click' 列是标签
val categoricalColumns = trainDf.select_dtypes(include = Seq("object")).columns
println(categoricalColumns)

// 记录训练集的行数，用于后续拆分
val trainRows = trainDf.length

// 合并训练集和测试集
val combinedDf = pd.concat(Seq(trainDf, testDf), ignore_index = true)

val labelEncoders = collection.mutable.Map[String, LabelEncoder]()
for col <- categoricalColumns do
  val le = new LabelEncoder()
  combinedDf(col) = le.fit_transform(combinedDf(col).astype[String])
  labelEncoders(col) = le

// 拆分回训练集和测试集
val newTrainDf = combinedDf.slice(0, trainRows)
val newTestDf = combinedDf.slice(trainRows, combinedDf.length)

println("数据预处理完成。")

// 划分特征和标签
val X = newTrainDf.drop(columns = Seq("click")).values.toTensor[Float32]
val y = newTrainDf("click").values.toTensor[Float32]

// 划分训练集和验证集
val Array(XTrain, XVal, yTrain, yVal) = train_test_split(X, y, test_size = 0.2, random_state = 42)

// 定义数据集类
class AvazuDataset(X: Tensor[Float32], y: Tensor[Float32]) extends Dataset[Tensor[Float32], Tensor[Float32]]:
  def len(): Int = X.shape(0)
  def apply(idx: Int): (Tensor[Float32], Tensor[Float32]) =
    (X(idx), y(idx))

// 定义 MoE 层
class MoE(numExperts: Int, dModel: Int, numClasses: Int) extends Module:
  val experts = ModuleList[Linear](Seq.fill(numExperts)(new Linear(dModel, numClasses)))
  val gate = new Linear(dModel, numExperts)

  def forward(x: Tensor[Float32]): Tensor[Float32] =
    val gateOutput = gate(x)
    val gateWeights = torch.softmax(gateOutput, dim = 1)
    val expertOutputs = torch.stack(experts.map(_(x)), dim = 1)
    val output = torch.sum(gateWeights.unsqueeze(-1) * expertOutputs, dim = 1)
    output

// 定义 Transformer MoE 推荐模型
class TransformerMoERecommender(inputDim: Int, dModel: Int, nhead: Int, numLayers: Int, numExperts: Int, numClasses: Int, dropout: Double) extends Module:
  val embedding = new Linear(inputDim, dModel)
  val transformerEncoder = new TransformerEncoder(
    new TransformerEncoderLayer(d_model = dModel, nhead = nhead, dropout = dropout, batch_first = true),
    num_layers = numLayers
  )
  val moe = new MoE(numExperts, dModel, numClasses)
  val dropoutLayer = new Dropout(dropout)

  def forward(x: Tensor[Float32]): Tensor[Float32] =
    var out = embedding(x)
    out = transformerEncoder(out)
    out = moe(out)
    torch.sigmoid(out).squeeze()

// 初始化数据集和数据加载器
println("初始化数据集和数据加载器。")
val trainDataset = new AvazuDataset(XTrain, yTrain)
val valDataset = new AvazuDataset(XVal, yVal)

val trainLoader = DataLoader(trainDataset, batch_size = 64, shuffle = true)
val valLoader = DataLoader(valDataset, batch_size = 64, shuffle = false)
println("初始化数据集和数据加载器完成。")

// 模型参数
val INPUT_DIM = XTrain.shape(1)
val D_MODEL = 128
val NHEAD = 4
val NUM_LAYERS = 2
val NUM_EXPERTS = 4
val NUM_CLASSES = 1
val DROPOUT = 0.5

// 初始化模型、损失函数和优化器
val model = new TransformerMoERecommender(INPUT_DIM, D_MODEL, NHEAD, NUM_LAYERS, NUM_EXPERTS, NUM_CLASSES, DROPOUT).to(device)
val criterion = new BCELoss()
val optimizer = new Adam(model.parameters(), lr = 0.001)

// 训练函数
def train(model: Module, dataloader: DataLoader[Tensor[Float32], Tensor[Float32]], optimizer: Optimizer, criterion: Loss[Float32]): Double =
  model.train()
  var totalLoss = 0.0
  var index = 0
  println("开始训练")
  for (X, y) <- dataloader do
    index += 1
    val XDevice = X.to(device)
    val yDevice = y.to(device)
    optimizer.zero_grad()
    val outputs = model(XDevice).asInstanceOf[Tensor[Float32]]
    val loss = criterion(outputs, yDevice)
    loss.backward()
    optimizer.step()
    totalLoss += loss.item()
    println(f"index $index total_loss $totalLoss loss: ${loss.item()}%.4f")
  totalLoss / dataloader.length

// 评估函数
def evaluate(model: Module, dataloader: DataLoader[Tensor[Float32], Tensor[Float32]], criterion: Loss[Float32]): Double =
  model.eval()
  var totalLoss = 0.0
  var index = 0
  println("开始评估")
  torch.no_grad {
    for (X, y) <- dataloader do
      index += 1
      val XDevice = X.to(device)
      val yDevice = y.to(device)
      val outputs = model(XDevice).asInstanceOf[Tensor[Float32]]
      val loss = criterion(outputs, yDevice)
      totalLoss += loss.item()
      println(f"index $index total_loss $totalLoss loss: ${loss.item()}%.4f")
  }
  totalLoss / dataloader.length

// 训练循环
val N_EPOCHS = 10
for epoch <- 1 to N_EPOCHS do
  println(f"Epoch $epoch/$N_EPOCHS")
  val trainLoss = train(model, trainLoader, optimizer, criterion)
  val valLoss = evaluate(model, valLoader, criterion)
  println(f'Epoch $epoch/$N_EPOCHS, Train Loss: $trainLoss%.4f, Val Loss: $valLoss%.4f')

println("Training finished.")
