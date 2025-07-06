
import scala.collection.mutable
import torch.*
import torch.nn.*
import torch.optim.*
import torch.utils.data.*

// 检查 GPU 是否可用
val device = if torch.cuda.isAvailable() then torch.device("cuda") else torch.device("cpu")
println(s"Using device: $device")

// 下载并解压 MovieLens 1M 数据集
val DATA_URL = "http://files.grouplens.org/datasets/movielens/ml-1m.zip"
val DATA_DIR = "ml-1m"
val DATA_PATH = Paths.get(DATA_DIR, "ratings.dat").toString
val ZIP_PATH = "ml-1m.zip"

if!Files.exists(Paths.get(DATA_DIR)) then
  println("Downloading MovieLens 1M dataset...")
  val in = requests.get(DATA_URL).inputStream()
  Files.copy(in, Paths.get(ZIP_PATH))
  in.close()

  println("Extracting dataset...")
  val zipFile = new java.util.zip.ZipFile(ZIP_PATH)
  val entries = zipFile.entries()
  while entries.hasMoreElements() do
    val entry = entries.nextElement()
    val entryPath = Paths.get(DATA_DIR, entry.getName)
    if entry.isDirectory() then
      Files.createDirectories(entryPath)
    else
      Files.copy(zipFile.getInputStream(entry), entryPath)
  zipFile.close()
  Files.delete(Paths.get(ZIP_PATH))

// 读取数据
println("pandas Reading data...")
val df = DataFrame.read_csv(DATA_PATH, sep = "::", engine = "python", header = None, names = Seq("userId", "movieId", "rating", "timestamp"))

// 对用户 ID 和电影 ID 进行编码
val userIds = df("userId").unique()
val movieIds = df("movieId").unique()

val userIdMap = mutable.Map[Any, Int]()
userIds.zipWithIndex.foreach { case (id, idx) => userIdMap(id) = idx }

val movieIdMap = mutable.Map[Any, Int]()
movieIds.zipWithIndex.foreach { case (id, idx) => movieIdMap(id) = idx }

df("userId") = df("userId").map(id => userIdMap(id))
df("movieId") = df("movieId").map(id => movieIdMap(id))

println("Data loaded train_test_split .")
// 划分训练集和测试集
val Array(trainData, testData) = train_test_split(df.toArray, test_size = 0.2, random_state = 42)
val trainDf = DataFrame.fromArray(trainData, columns = df.columns)
val testDf = DataFrame.fromArray(testData, columns = df.columns)

// 定义数据集类
class MovieLensDataset(data: DataFrame) extends Dataset[Tensor[Long], (Tensor[Long], Tensor[Float32])]:
  val userIds = torch.tensor(data("userId").values.toArray.map(_.asInstanceOf[Int]), dtype = torch.long)
  val movieIds = torch.tensor(data("movieId").values.toArray.map(_.asInstanceOf[Int]), dtype = torch.long)
  val ratings = torch.tensor(data("rating").values.toArray.map(_.asInstanceOf[Float]), dtype = torch.float32)

  def len(): Int = userIds.size(0)

  def apply(idx: Int): (Tensor[Long], Tensor[Long], Tensor[Float32]) =
    (userIds(idx), movieIds(idx), ratings(idx))

// 定义 MoE 层
class MoE(numExperts: Int, dModel: Int, numClasses: Int) extends Module:
  val experts = ModuleList[Linear](Seq.fill(numExperts)(new Linear(dModel, numClasses)))
  val gate = new Linear(dModel, numExperts)

  def forward(x: Tensor[Float32]): Tensor[Float32] =
    val gateOutput = gate(x)
    val gateWeights = torch.softmax(gateOutput, dim = 1)
    val expertOutputs = torch.stack(experts.map(_(x)), dim = 1)
    torch.sum(gateWeights.unsqueeze(-1) * expertOutputs, dim = 1)

// 定义 Transformer MoE 推荐模型
class TransformerMoERecommender(numUsers: Int, numMovies: Int, dModel: Int, nhead: Int, numLayers: Int, numExperts: Int, dropout: Double) extends Module:
  val userEmbedding = new Embedding(numUsers, dModel)
  val movieEmbedding = new Embedding(numMovies, dModel)
  val transformerEncoder = new TransformerEncoder(
    new TransformerEncoderLayer(d_model = dModel, nhead = nhead, dropout = dropout),
    num_layers = numLayers
  )
  val moe = new MoE(numExperts, dModel, 1)
  val dropoutLayer = new Dropout(dropout)

  def forward(userIds: Tensor[Long], movieIds: Tensor[Long]): Tensor[Float32] =
    val userEmbed = userEmbedding(userIds).asInstanceOf[Tensor[Float32]]
    val movieEmbed = movieEmbedding(movieIds).asInstanceOf[Tensor[Float32]]
    val combinedEmbed = userEmbed + movieEmbed
    val combinedEmbedSeq = combinedEmbed.unsqueeze(0) // 添加序列维度
    var output = transformerEncoder(combinedEmbedSeq)
    output = output.squeeze(0)
    output = moe(output)
    output.squeeze()

// 初始化数据集和数据加载器
println("Initializing dataset and dataloader...")
val trainDataset = new MovieLensDataset(trainDf)
val testDataset = new MovieLensDataset(testDf)

val trainLoader = DataLoader(trainDataset, batchSize = 64, shuffle = true)
val testLoader = DataLoader(testDataset, batchSize = 64, shuffle = false)
println("Dataset and dataloader initialized successfully.")

// 模型参数
val NUM_USERS = userIdMap.size
val NUM_MOVIES = movieIdMap.size
val D_MODEL = 128
val NHEAD = 4
val NUM_LAYERS = 2
val NUM_EXPERTS = 4
val DROPOUT = 0.5

// 初始化模型、损失函数和优化器
val model = new TransformerMoERecommender(NUM_USERS, NUM_MOVIES, D_MODEL, NHEAD, NUM_LAYERS, NUM_EXPERTS, DROPOUT).to(device)
val criterion = new MSELoss()
val optimizer = new Adam(model.parameters(), lr = 0.001)

// 训练函数
def train(model: Module, dataloader: DataLoader[Tensor[Long], (Tensor[Long], Tensor[Float32])], optimizer: Optimizer, criterion: Loss[Float32]): Double =
  model.train()
  var totalLoss = 0.0
  println("Training model...")
  for (userIds, movieIds, ratings) <- dataloader do
    val userIdsDevice = userIds.to(device)
    val movieIdsDevice = movieIds.to(device)
    val ratingsDevice = ratings.to(device)
    optimizer.zeroGrad()
    val outputs = model(userIdsDevice, movieIdsDevice).asInstanceOf[Tensor[Float32]]
    val loss = criterion(outputs, ratingsDevice)
    loss.backward()
    optimizer.step()
    totalLoss += loss.item()
  totalLoss / dataloader.length

// 评估函数
def evaluate(model: Module, dataloader: DataLoader[Tensor[Long], (Tensor[Long], Tensor[Float32])], criterion: Loss[Float32]): Double =
  model.eval()
  var totalLoss = 0.0
  torch.no_grad {
    println("Evaluating model...")
    for (userIds, movieIds, ratings) <- dataloader do
      val userIdsDevice = userIds.to(device)
      val movieIdsDevice = movieIds.to(device)
      val ratingsDevice = ratings.to(device)
      val outputs = model(userIdsDevice, movieIdsDevice).asInstanceOf[Tensor[Float32]]
      val loss = criterion(outputs, ratingsDevice)
      totalLoss += loss.item()
  }
  totalLoss / dataloader.length

// 训练循环
val N_EPOCHS = 10
for epoch <- 1 to N_EPOCHS do
  println(s"Epoch $epoch/$N_EPOCHS")
  val trainLoss = train(model, trainLoader, optimizer, criterion)
  val testLoss = evaluate(model, testLoader, criterion)
  println(f"Epoch $epoch/$N_EPOCHS, Train Loss: $trainLoss%.4f, Test Loss: $testLoss%.4f")

println("Training finished.")
