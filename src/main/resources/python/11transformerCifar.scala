import torch.*
import torch.nn.*
import torch.optim.*
import storch.vision.datasets.CIFAR100
import storch.vision.transforms.*
import torch.utils.data.*

// 检查 GPU 是否可用
val device = if torch.cuda.isAvailable() then torch.device("cuda") else torch.device("cpu")
println(s"Using device: $device")

// 数据预处理
val transform = Compose(
  Seq(
    Resize(32, 32),
    ToTensor(),
    Normalize(Array(0.5, 0.5, 0.5), Array(0.5, 0.5, 0.5))
  )
)

// 加载 CIFAR - 100 数据集
val trainDataset = CIFAR100(root = "./data", train = true, download = true, transform = transform)
val trainLoader = DataLoader(trainDataset, batchSize = 64, shuffle = true)

// 定义 Transformer 模型
class TransformerClassifier(inputDim: Int, dModel: Int, nhead: Int, numLayers: Int, numClasses: Int) extends Module:
  val embedding = new Linear(inputDim, dModel)
  val positionalEncoding = new PositionalEncoding(dModel)
  val transformerEncoder = new TransformerEncoder(
    new TransformerEncoderLayer(d_model = dModel, nhead = nhead),
    num_layers = numLayers
  )
  val transformerDecoder = new TransformerDecoder(
    new TransformerDecoderLayer(d_model = dModel, nhead = nhead),
    num_layers = numLayers
  )
  val fc = new Linear(dModel, numClasses)

  def forward(src: Tensor[Float32]): Tensor[Float32] =
    val flattenedSrc = src.view(src.size(0), -1) // 展平图像
    val embeddedSrc = embedding(flattenedSrc)
    val srcWithSeqDim = embeddedSrc.unsqueeze(0) // 添加序列维度 [1, batch_size, d_model]
    val srcWithPos = positionalEncoding(srcWithSeqDim)
    val memory = transformerEncoder(srcWithPos)
    val tgt = torch.zeros_like(memory) // 简单的目标输入
    var output = transformerDecoder(tgt, memory)
    output = output.squeeze(0) // 移除序列维度
    output = fc(output)
    output

class PositionalEncoding(dModel: Int, maxLen: Int = 5000) extends Module:
  val pe = torch.zeros(maxLen, dModel)
  val position = torch.arange(0, maxLen, dtype = torch.float32).unsqueeze(1)
  val divTerm = torch.exp(torch.arange(0, dModel, 2).float32() * (-torch.log(torch.tensor(10000.0))) / dModel)
  pe(::, 0::2) = torch.sin(position * divTerm)
  pe(::, 1::2) = torch.cos(position * divTerm)
  registerBuffer("pe", pe.unsqueeze(1))

  def forward(x: Tensor[Float32]): Tensor[Float32] =
    x + pe.slice(0, x.size(0), ::, ::)

// 模型参数
val inputDim = 3 * 32 * 32 // CIFAR - 100 图像大小
val dModel = 128
val nhead = 4
val numLayers = 2
val numClasses = 100

// 初始化模型、损失函数和优化器
val model = new TransformerClassifier(inputDim, dModel, nhead, numLayers, numClasses).to(device)
val criterion = new CrossEntropyLoss()
val optimizer = new Adam(model.parameters(), lr = 0.001)

// 训练模型
val numEpochs = 5
for epoch <- 1 to numEpochs do
  model.train()
  var totalLoss = 0.0
  for (images, labels) <- trainLoader do
    val imagesDevice = images.to(device)
    val labelsDevice = labels.to(device)

    optimizer.zeroGrad()
    val outputs = model(imagesDevice).asInstanceOf[Tensor[Float32]]
    val loss = criterion(outputs, labelsDevice)
    loss.backward()
    optimizer.step()

    totalLoss += loss.item()

  val avgLoss = totalLoss / trainLoader.length
  println(f"Epoch $epoch/$numEpochs, Loss: $avgLoss%.4f")

println("Training finished.")
