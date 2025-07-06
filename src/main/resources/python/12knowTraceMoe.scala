import torch.*
import torch.nn.*
import torch.optim.*
import torch.utils.data.*

import scala.collection.mutable.ArrayBuffer

// 检查 GPU 是否可用
val device = if torch.cuda.isAvailable() then torch.device("cuda") else torch.device("cpu")
println(s"Using device: $device")

// 数据加载与预处理
def loadAndPreprocessData(filePath: String): Tensor[Float32] =
  val data = DataFrame.read_csv(Paths.get(filePath))
  // 假设前 20 列是特征，第 21 列是技能 ID，第 22 列是是否正确，第 23 列是时间戳，第 24 列是耗时
  val featureColumns = data.columns.take(20)
  val skillColumn = data.columns(20)
  val correctColumn = data.columns(21)
  val timestampColumn = data.columns(22)
  val durationColumn = data.columns(23)

  val userIds = data("user_id").unique()
  val sequences = ArrayBuffer[Tensor[Float32]]()
  for userId <- userIds do
    val userData = data.filter(data("user_id") === userId)
    val sortedUserData = userData.sort_values(by = Seq(timestampColumn))

    val features = sortedUserData(featureColumns).values.toTensor[Float32]
    val skills = sortedUserData(skillColumn).values.toTensor[Long].toFloat32()
    val corrects = sortedUserData(correctColumn).values.toTensor[Long].toFloat32()
    val timestamps = sortedUserData(timestampColumn).values.toTensor[Float32]
    val durations = sortedUserData(durationColumn).values.toTensor[Float32]

    val sequence = torch.cat(
      Seq(
        features,
        skills.unsqueeze(1),
        corrects.unsqueeze(1),
        timestamps.unsqueeze(1),
        durations.unsqueeze(1)
      ),
      dim = 1
    )
    sequences.append(sequence)

  // 填充序列
  val maxLength = sequences.map(_.size(0)).max
  val paddedSequences = sequences.map { seq =>
    val padding = torch.zeros(maxLength - seq.size(0), seq.size(1), dtype = torch.float32)
    torch.cat(Seq(seq, padding), dim = 0)
  }
  torch.stack(paddedSequences)

// 自定义数据集类
class KnowledgeTracingDataset(sequences: Tensor[Float32]) extends Dataset[Tensor[Float32], (Tensor[Long], Tensor[Long], Tensor[Float32], Tensor[Float32])]:
  def len(): Int = sequences.size(0)

  def apply(idx: Int): (Tensor[Float32], Tensor[Long], Tensor[Long], Tensor[Float32], Tensor[Float32]) =
    val seq = sequences(idx)
    val features = seq(::, 0::20)
    val skills = seq(::, 20).toLong()
    val corrects = seq(::, 21).toLong()
    val timestamps = seq(::, 22)
    val durations = seq(::, 23)
    (features, skills, corrects, timestamps, durations)

// 定义 IRT 模块
class IRTModule(numSkills: Int) extends Module:
  val ability = new Embedding(1, 1) // 全局能力参数
  val difficulty = new Embedding(numSkills, 1)
  val discrimination = new Embedding(numSkills, 1)

  def forward(skills: Tensor[Long]): Tensor[Float32] =
    val abilityVal = ability(torch.zeros(skills.size(0), dtype = torch.long, device = device))
    val difficultyVal = difficulty(skills)
    val discriminationVal = discrimination(skills)
    val logits = discriminationVal * (abilityVal - difficultyVal)
    torch.sigmoid(logits).squeeze(-1)

// 定义专家网络
class Expert(inputDim: Int, dModel: Int) extends Module:
  val fc1 = new Linear(inputDim, dModel)
  val fc2 = new Linear(dModel, dModel)

  def forward(x: Tensor[Float32]): Tensor[Float32] =
    val out = nn.functional.relu(fc1(x))
    fc2(out)

// 定义门控网络
class GatingNetwork(inputDim: Int, numExperts: Int) extends Module:
  val fc = new Linear(inputDim, numExperts)

  def forward(x: Tensor[Float32]): Tensor[Float32] =
    nn.functional.softmax(fc(x), dim = 1)

// 定义 MoE 模块
class MoE(inputDim: Int, dModel: Int, numExperts: Int) extends Module:
  val experts = ModuleList[Expert](Seq.fill(numExperts)(new Expert(inputDim, dModel)))
  val gatingNetwork = new GatingNetwork(inputDim, numExperts)

  def forward(x: Tensor[Float32]): Tensor[Float32] =
    val gates = gatingNetwork(x)
    val expertOutputs = experts.map(_(x))
    val stackedOutputs = torch.stack(expertOutputs, dim = 1)
    torch.sum(gates.unsqueeze(-1) * stackedOutputs, dim = 1)

// 定义 Transformer 块
class TransformerBlock(inputDim: Int, dModel: Int, numExperts: Int, dFf: Int, nhead: Int, dropout: Double) extends Module:
  val moe = new MoE(inputDim, dModel, numExperts)
  val norm1 = new LayerNorm(dModel)
  val dropout1 = new Dropout(dropout)

  val selfAttn = new MultiheadAttention(dModel, nhead)
  val norm2 = new LayerNorm(dModel)
  val dropout2 = new Dropout(dropout)

  val feedForward = nn.Sequential(
    new Linear(dModel, dFf),
    new ReLU(),
    new Linear(dFf, dModel)
  )
  val norm3 = new LayerNorm(dModel)
  val dropout3 = new Dropout(dropout)

  def forward(x: Tensor[Float32]): Tensor[Float32] =
    val moeOutput = moe(x)
    var out = norm1(x + dropout1(moeOutput))

    val (attnOutput, _) = selfAttn(out, out, out)
    out = norm2(out + dropout2(attnOutput))

    val ffOutput = feedForward(out)
    norm3(out + dropout3(ffOutput))

// 定义知识追踪模型
class KnowledgeTracingModel(numSkills: Int, inputDim: Int, dModel: Int, numExperts: Int, dFf: Int, nhead: Int, numLayers: Int, dropout: Double) extends Module:
  val skillEmbedding = new Embedding(numSkills, inputDim / 4)
  val featureProj = new Linear(20, inputDim / 4)
  val timeEmbedding = new Linear(1, inputDim / 4)
  val durationEmbedding = new Linear(1, inputDim / 4)
  val irtModule = new IRTModule(numSkills)
  val transformerBlocks = ModuleList[TransformerBlock](
    Seq.fill(numLayers)(new TransformerBlock(inputDim, dModel, numExperts, dFf, nhead, dropout))
  )
  val fc = new Linear(dModel, 1)
  val sigmoid = new Sigmoid()

  def forward(features: Tensor[Float32], skills: Tensor[Long], timestamps: Tensor[Float32], durations: Tensor[Float32]): Tensor[Float32] =
    val featureEmbed = featureProj(features)
    val skillEmbed = skillEmbedding(skills)
    val timeEmbed = timeEmbedding(timestamps.unsqueeze(-1))
    val durationEmbed = durationEmbedding(durations.unsqueeze(-1))

    val x = torch.cat(Seq(featureEmbed, skillEmbed, timeEmbed, durationEmbed), dim = -1)

    val irtProb = irtModule(skills)

    var out = x
    for block <- transformerBlocks do
      out = block(out)

    val output = fc(out)
    val outputProb = sigmoid(output).squeeze(-1)

    // 结合 IRT 概率
    0.5 * outputProb + 0.5 * irtProb

// 训练函数
def train(model: Module, device: Device, trainLoader: DataLoader[Tensor[Float32], (Tensor[Long], Tensor[Long], Tensor[Float32], Tensor[Float32])], criterion: Loss[Float32], optimizer: Optimizer, epoch: Int): Unit =
  model.train()
  var totalLoss = 0.0
  for (features, skills, corrects, timestamps, durations) <- trainLoader do
    val inputsFeatures = features(::, 0 :: -1, ::)
    val inputsSkills = skills(::, 0 :: -1)
    val inputsTimestamps = timestamps(::, 0 :: -1)
    val inputsDurations = durations(::, 0 :: -1)
    val targets = corrects(::, 1::).toFloat32()

    val featuresDevice = inputsFeatures.to(device)
    val skillsDevice = inputsSkills.to(device)
    val timestampsDevice = inputsTimestamps.to(device)
    val durationsDevice = inputsDurations.to(device)
    val targetsDevice = targets.to(device)

    optimizer.zeroGrad()
    val outputs = model(featuresDevice, skillsDevice, timestampsDevice, durationsDevice).asInstanceOf[Tensor[Float32]]
    val loss = criterion(outputs, targetsDevice)
    loss.backward()
    optimizer.step()

    totalLoss += loss.item()

  val avgLoss = totalLoss / trainLoader.length
  println(f"Epoch ${epoch + 1}, Loss: $avgLoss%.4f")

// 主函数
@main def main(): Unit =
  val filePath = "your_dataset.csv" // 替换为实际的数据文件路径
  val sequences = loadAndPreprocessData(filePath)
  val numSkills = sequences(::, ::, 20).max().toInt() + 1

  val Array(trainSequences, testSequences) = train_test_split(sequences, test_size = 0.2, random_state = 42)

  val trainDataset = new KnowledgeTracingDataset(trainSequences)
  val testDataset = new KnowledgeTracingDataset(testSequences)

  val trainLoader = DataLoader(trainDataset, batchSize = 32, shuffle = true)
  val testLoader = DataLoader(testDataset, batchSize = 32, shuffle = false)

  // 模型参数
  val inputDim = 128
  val dModel = 256
  val numExperts = 4
  val dFf = 512
  val nhead = 4
  val numLayers = 2
  val dropout = 0.1

  val model = new KnowledgeTracingModel(numSkills, inputDim, dModel, numExperts, dFf, nhead, numLayers, dropout).to(device)
  val criterion = new BCELoss()
  val optimizer = new Adam(model.parameters(), lr = 0.001)

  val numEpochs = 10
  for epoch <- 0 until numEpochs do
    train(model, device, trainLoader, criterion, optimizer, epoch)
