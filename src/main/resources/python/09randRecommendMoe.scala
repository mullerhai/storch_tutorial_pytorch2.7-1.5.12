import torch.*
import torch.nn.*
import torch.optim.*
import torch.utils.data.*

// 检查 GPU 是否可用
val device = if torch.cuda.isAvailable() then torch.device("cuda") else torch.device("cpu")
println(s"Using device: $device")

// 模拟淘宝电商数据集
class TaobaoDataset(numUsers: Int = 1000, numItems: Int = 500, numSamples: Int = 10000) extends Dataset[Tensor[Long], (Tensor[Long], Tensor[Float32])]:
  val userIds = torch.randint(0, numUsers, numSamples).to(torch.long)
  val itemIds = torch.randint(0, numItems, numSamples).to(torch.long)
  val labels = torch.randint(0, 2, numSamples).to(torch.float32)

  def len(): Int = userIds.size(0)

  def apply(idx: Int): (Tensor[Long], Tensor[Long], Tensor[Float32]) =
    (userIds(idx), itemIds(idx), labels(idx))

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

  def forward(userIds: Tensor[Long], itemIds: Tensor[Long]): Tensor[Float32] =
    val userEmbed = userEmbedding(userIds).asInstanceOf[Tensor[Float32]]
    val itemEmbed = itemEmbedding(itemIds).asInstanceOf[Tensor[Float32]]
    val combinedEmbed = userEmbed + itemEmbed
    val combinedEmbedSeq = combinedEmbed.unsqueeze(0) // 添加序列维度
    var output = transformerEncoder(combinedEmbedSeq)
    output = output.squeeze(0)
    output = moe(output)
    torch.sigmoid(output).squeeze()

// 初始化数据集和数据加载器
val trainDataset = new TaobaoDataset()
val trainLoader = DataLoader(trainDataset, batchSize = 64, shuffle = true)

// 模型参数
val NUM_USERS = 1000
val NUM_ITEMS = 500
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
def train(model: Module, dataloader: DataLoader[Tensor[Long], (Tensor[Long], Tensor[Float32])], optimizer: Optimizer, criterion: Loss[Float32]): Double =
  model.train()
  var totalLoss = 0.0
  for (userIds, itemIds, labels) <- dataloader do
    val userIdsDevice = userIds.to(device)
    val itemIdsDevice = itemIds.to(device)
    val labelsDevice = labels.to(device)
    optimizer.zeroGrad()
    val outputs = model(userIdsDevice, itemIdsDevice).asInstanceOf[Tensor[Float32]]
    val loss = criterion(outputs, labelsDevice)
    loss.backward()
    optimizer.step()
    totalLoss += loss.item()
  totalLoss / dataloader.length

// 训练循环
val N_EPOCHS = 10
for epoch <- 1 to N_EPOCHS do
  val trainLoss = train(model, trainLoader, optimizer, criterion)
  println(f"Epoch $epoch/$N_EPOCHS, Train Loss: $trainLoss%.4f")

println("Training finished.")
