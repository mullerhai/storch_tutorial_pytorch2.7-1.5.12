import torch.*
import torch.nn.*
import torch.optim.*
import torch.utils.data.*

// 检查 GPU 是否可用
val device = if torch.cuda.isAvailable() then torch.device("cuda") else torch.device("cpu")
println(s"Using device: $device")

// 模拟用户、航班和景点数据
val numUsers = 1000
val numFlights = 500
val numAttractions = 300

// 模拟用户与航班、景点的交互数据
val userFlightInteractions = np.random.randint(0, 2, size = Array(numUsers, numFlights)).toTensor[Float32]
val userAttractionInteractions = np.random.randint(0, 2, size = Array(numUsers, numAttractions)).toTensor[Float32]

// 生成训练数据
val allData = mutable.ArrayBuffer[(Int, Int, Float, Int)]()
println("Generating training data...")
for userID <- 0 until numUsers do
  println(s"Generating data for user $userID...")
  for flightID <- 0 until numFlights do
    allData.append((userID, flightID, userFlightInteractions(userID, flightID).item(), 0)) // 0 表示航班
  for attractionID <- 0 until numAttractions do
    allData.append((userID, attractionID + numFlights, userAttractionInteractions(userID, attractionID).item(), 1)) // 1 表示景点
println("Training data generated.")

// 划分训练集和测试集
val (trainData, testData) = train_test_split(allData.toArray, test_size = 0.2, random_state = 42)

// 定义数据集类
class FlightAttractionDataset(data: Array[(Int, Int, Float, Int)]) extends Dataset[Tensor[Long], (Tensor[Long], Tensor[Float32], Tensor[Long])]:
  def len(): Int = data.length

  def apply(idx: Int): (Tensor[Long], Tensor[Long], Tensor[Float32], Tensor[Long]) =
    val (userID, itemID, label, itemType) = data(idx)
    (
      torch.tensor(userID, dtype = torch.long),
      torch.tensor(itemID, dtype = torch.long),
      torch.tensor(label, dtype = torch.float32),
      torch.tensor(itemType, dtype = torch.long)
    )

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
class TransformerMoERecommender(numUsers: Int, numItems: Int, dModel: Int, nhead: Int, numLayers: Int, numExperts: Int, dropout: Double) extends Module:
  val userEmbedding = new Embedding(numUsers, dModel)
  val itemEmbedding = new Embedding(numItems, dModel)
  val transformerEncoder = new TransformerEncoder(
    new TransformerEncoderLayer(d_model = dModel, nhead = nhead, dropout = dropout),
    num_layers = numLayers
  )
  val moe = new MoE(numExperts, dModel, 1)
  val dropoutLayer = new Dropout(dropout)

  def forward(userIDs: Tensor[Long], itemIDs: Tensor[Long]): Tensor[Float32] =
    val userEmbed = userEmbedding(userIDs).asInstanceOf[Tensor[Float32]]
    val itemEmbed = itemEmbedding(itemIDs).asInstanceOf[Tensor[Float32]]
    val combinedEmbed = userEmbed + itemEmbed
    val combinedEmbedSeq = combinedEmbed.unsqueeze(0) // 添加序列维度
    var output = transformerEncoder(combinedEmbedSeq)
    output = output.squeeze(0)
    output = moe(output)
    torch.sigmoid(output).squeeze()

// 初始化数据集和数据加载器
println("Initializing dataset and dataloader...")
val trainDataset = new FlightAttractionDataset(trainData)
val testDataset = new FlightAttractionDataset(testData)
val trainLoader = DataLoader(trainDataset, batchSize = 64, shuffle = true)
val testLoader = DataLoader(testDataset, batchSize = 64, shuffle = false)
println("Dataset and dataloader initialized. completed")

// 模型参数
val NUM_USERS = numUsers
val NUM_ITEMS = numFlights + numAttractions
val D_MODEL = 128
val NHEAD = 4
val NUM_LAYERS = 2
val NUM_EXPERTS = 4
val DROPOUT = 0.5

// 初始化模型、损失函数和优化器
val model = new TransformerMoERecommender(NUM_USERS, NUM_ITEMS, D_MODEL, NHEAD, NUM_LAYERS, NUM_EXPERTS, DROPOUT).to(device)
val criterion = new BCELoss()
val optimizer = new Adam(model.parameters(), lr = 0.001)

// 训练函数
def train(model: Module, dataloader: DataLoader[Tensor[Long], (Tensor[Long], Tensor[Float32], Tensor[Long])], optimizer: Optimizer, criterion: Loss[Float32]): Double =
  model.train()
  var totalLoss = 0.0
  var userIndex = 0
  println("Starting training...")
  for (userIDs, itemIDs, labels, _) <- dataloader do
    val userIDsDevice = userIDs.to(device)
    val itemIDsDevice = itemIDs.to(device)
    val labelsDevice = labels.to(device)
    optimizer.zeroGrad()
    val outputs = model(userIDsDevice, itemIDsDevice).asInstanceOf[Tensor[Float32]]
    val loss = criterion(outputs, labelsDevice)
    loss.backward()
    optimizer.step()
    userIndex += 1
    println(s"Batch loss: ${loss.item()} for user $userIndex")
    totalLoss += loss.item()
  totalLoss / dataloader.length

// 评估函数
def evaluate(model: Module, dataloader: DataLoader[Tensor[Long], (Tensor[Long], Tensor[Float32], Tensor[Long])], criterion: Loss[Float32]): Double =
  model.eval()
  var totalLoss = 0.0
  var index = 0
  torch.no_grad {
    println("Starting evaluation...")
    for (userIDs, itemIDs, labels, _) <- dataloader do
      index += 1
      val userIDsDevice = userIDs.to(device)
      val itemIDsDevice = itemIDs.to(device)
      val labelsDevice = labels.to(device)
      val outputs = model(userIDsDevice, itemIDsDevice).asInstanceOf[Tensor[Float32]]
      val loss = criterion(outputs, labelsDevice)
      totalLoss += loss.item()
      println(s"evaluation Batch loss: ${loss.item()} for user index  $index")
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
