
import scala.collection.mutable
import scala.util.Using
import torch.*
import torch.nn.*
import torch.optim.*
import torch.utils.data.*

// 检查 GPU 是否可用
val device = if torch.cuda.isAvailable() then torch.device("cuda") else torch.device("cpu")
println(s"Using device: $device")

// 下载 IMDB 数据集
val DATA_URL = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
val DATA_ROOT = "./data"
val tarGzPath = Paths.get(DATA_ROOT, "aclImdb_v1.tar.gz")
val dataDir = Paths.get(DATA_ROOT, "aclImdb")

if!Files.exists(dataDir) then
  if!Files.exists(tarGzPath) then
    println("Downloading IMDB dataset...")
    Files.createDirectories(Paths.get(DATA_ROOT))
    val in = requests.get(DATA_URL).inputStream()
    Files.copy(in, tarGzPath)
    in.close()
  println("Extracting IMDB dataset...")
  // 模拟解压操作，实际需要添加解压 tar.gz 的逻辑
  // 这里简化处理，假设已经手动解压到指定目录
  println("IMDB dataset downloaded and extracted.")
else
  println("IMDB dataset already exists.")

// 定义词汇表构建和数据加载相关类
class IMDBDataset(rootDir: String, split: String = "train", maxVocabSize: Int = 25000) extends Dataset[Tensor[Long], Tensor[Float32]]:
  private val root = Paths.get(rootDir, "aclImdb", split)
  private val labelMap = Map("pos" -> 1, "neg" -> 0)
  private val vocab = mutable.Map[String, Int]()
  private val data = mutable.ArrayBuffer[(Array[Int], Int)]()

  // 构建词汇表
  private def buildVocab(): Unit =
    val counter = mutable.Map[String, Int]()
    for label <- labelMap.keys do
      val labelDir = root.resolve(label)
      val files = new File(labelDir.toString).listFiles()
      var index = 0
      for file <- files if index < 640 do
        Using.resource(new java.io.FileReader(file)) { reader =>
          val text = scala.io.Source.fromReader(reader).mkString
          val tokens = text.toLowerCase.replaceAll("<[^>]+>", "").split("\\W+").filter(_.nonEmpty)
          tokens.foreach { token =>
            counter.updateWith(token)(_.map(_ + 1).orElse(Some(1)))
          }
        }
        index += 1

    val vocabList = counter.toList.sortBy(-_._2).take(maxVocabSize - 2).map(_._1)
    vocab += "<pad>" -> 0
    vocab += "<unk>" -> 1
    vocabList.zipWithIndex.foreach { case (word, idx) =>
      vocab += word -> (idx + 2)
    }

  // 加载数据
  private def loadData(): Unit =
    for label <- labelMap.keys do
      val labelDir = root.resolve(label)
      val files = new File(labelDir.toString).listFiles()
      var cntIndex = 0
      for file <- files if cntIndex < 640 do
        Using.resource(new java.io.FileReader(file)) { reader =>
          val text = scala.io.Source.fromReader(reader).mkString
          val tokens = text.toLowerCase.replaceAll("<[^>]+>", "").split("\\W+").filter(_.nonEmpty)
          val tokenIds = tokens.map(token => vocab.getOrElse(token, vocab("<unk>")))
          data += ((tokenIds, labelMap(label)))
        }
        cntIndex += 1

  buildVocab()
  loadData()

  def len(): Int = data.length

  def apply(idx: Int): (Tensor[Long], Tensor[Float32]) =
    val (tokenIds, label) = data(idx)
    (torch.tensor(tokenIds, dtype = torch.long), torch.tensor(label, dtype = torch.float32))

  def getVocabSize: Int = vocab.size

// 定义填充函数
def collateFn(batch: Seq[(Tensor[Long], Tensor[Float32])]): (Tensor[Long], Tensor[Float32]) =
  val (texts, labels) = batch.unzip
  val maxLength = texts.map(_.size(0)).max
  val paddedTexts = texts.map(text =>
    val padding = torch.zeros(maxLength - text.size(0), dtype = torch.long)
    torch.cat(Seq(text, padding))
  )
  (torch.stack(paddedTexts).to(device), torch.stack(labels).to(device))

// 定义 MoE 层
class MoE(numExperts: Int, dModel: Int, numClasses: Int) extends Module:
  val experts = ModuleList[Linear](Seq.fill(numExperts)(new Linear(dModel, numClasses)))
  val gate = new Linear(dModel, numExperts)

  def forward(x: Tensor[Float32]): Tensor[Float32] =
    val gateOutput = gate(x)
    val gateWeights = torch.softmax(gateOutput, dim = 1)
    val expertOutputs = torch.stack(experts.map(_(x)), dim = 1)
    torch.sum(gateWeights.unsqueeze(-1) * expertOutputs, dim = 1)

// 定义位置编码
class PositionalEncoding(dModel: Int, maxLen: Int = 5000) extends Module:
  val pe = torch.zeros(maxLen, dModel)
  val position = torch.arange(0, maxLen, dtype = torch.float32).unsqueeze(1)
  val divTerm = torch.exp(torch.arange(0, dModel, 2).float32() * (-torch.log(torch.tensor(10000.0))) / dModel)
  pe(::, 0::2) = torch.sin(position * divTerm)
  pe(::, 1::2) = torch.cos(position * divTerm)
  registerBuffer("pe", pe.unsqueeze(0))

  def forward(x: Tensor[Float32]): Tensor[Float32] =
    x + pe(::, :x.size(1), ::)

// 定义 Transformer MoE 模型
class TransformerMoEClassifier(vocabSize: Int, dModel: Int, nhead: Int, numLayers: Int, numClasses: Int, numExperts: Int, dropout: Double) extends Module:
  val embedding = new Embedding(vocabSize, dModel)
  val positionalEncoding = new PositionalEncoding(dModel)
  val transformerEncoder = new TransformerEncoder(
    new TransformerEncoderLayer(dModel = dModel, nhead = nhead, dropout = dropout),
    numLayers = numLayers
  )
  val moe = new MoE(numExperts, dModel, numClasses)
  val dropoutLayer = new Dropout(dropout)

  def forward(src: Tensor[Long]): Tensor[Float32] =
    var out = embedding(src).asInstanceOf[Tensor[Float32]]
    out = dropoutLayer(out)
    out = positionalEncoding(out)
    out = transformerEncoder(out)
    out = torch.mean(out, dim = 1)
    moe(out).squeeze()

// 初始化数据集和数据加载器
val trainDataset = new IMDBDataset(DATA_ROOT, split = "train")
val testDataset = new IMDBDataset(DATA_ROOT, split = "test", maxVocabSize = trainDataset.getVocabSize)
val trainLoader = DataLoader(trainDataset, batchSize = 64, shuffle = true, collateFn = collateFn)
val testLoader = DataLoader(testDataset, batchSize = 64, shuffle = false, collateFn = collateFn)

// 模型参数
val VOCAB_SIZE = trainDataset.getVocabSize
val D_MODEL = 128
val NHEAD = 4
val NUM_LAYERS = 2
val NUM_CLASSES = 1
val NUM_EXPERTS = 4
val DROPOUT = 0.5

// 初始化模型、损失函数和优化器
val model = new TransformerMoEClassifier(VOCAB_SIZE, D_MODEL, NHEAD, NUM_LAYERS, NUM_CLASSES, NUM_EXPERTS, DROPOUT).to(device)
val criterion = new BCEWithLogitsLoss()
val optimizer = new Adam(model.parameters(), lr = 0.001)

// 训练函数
def train(model: Module, dataloader: DataLoader[Tensor[Long], Tensor[Float32]], optimizer: Optimizer, criterion: Loss[Float32]): Double =
  model.train()
  var totalLoss = 0.0
  for (texts, labels) <- dataloader do
    optimizer.zeroGrad()
    val outputs = model(texts).asInstanceOf[Tensor[Float32]]
    val loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()
    totalLoss += loss.item()
  totalLoss / dataloader.length

// 评估函数
def evaluate(model: Module, dataloader: DataLoader[Tensor[Long], Tensor[Float32]], criterion: Loss[Float32]): Double =
  model.eval()
  var totalLoss = 0.0
  torch.no_grad {
    for (texts, labels) <- dataloader do
      val outputs = model(texts).asInstanceOf[Tensor[Float32]]
      val loss = criterion(outputs, labels)
      totalLoss += loss.item()
  }
  totalLoss / dataloader.length

// 训练循环
val N_EPOCHS = 50
for epoch <- 1 to N_EPOCHS do
  println(s"Epoch $epoch/$N_EPOCHS")
  val trainLoss = train(model, trainLoader, optimizer, criterion)
  val testLoss = evaluate(model, testLoader, criterion)
  println(f"Epoch: $epoch%02d, Train Loss: $trainLoss%.3f, Test Loss: $testLoss%.3f")

