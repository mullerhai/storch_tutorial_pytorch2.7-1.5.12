//import torch.*
//import torch.nn.*
//import torch.optim.*
//import torch.utils.data.*
//
//// 检查 GPU 是否可用
//val device = if torch.cuda.isAvailable() then torch.device("cuda") else torch.device("cpu")
//println(s"Using device: $device")
//
//// 数据预处理
//val transform = Compose(
//  Seq(
//    ToTensor(),
//    Normalize(Array(0.1307), Array(0.3081))
//  )
//)
//
//// 加载 MNIST 数据集
//val trainDataset = MNIST(root = "./data", train = true, download = true, transform = transform)
//val trainLoader = DataLoader(trainDataset, batchSize = 64, shuffle = true)
//
//val testDataset = MNIST(root = "./data", train = false, download = true, transform = transform)
//val testLoader = DataLoader(testDataset, batchSize = 64, shuffle = false)
//
//// 定义专家网络
//class Expert(inputDim: Int, dModel: Int) extends Module:
//  val fc1 = new Linear(inputDim, dModel)
//  val fc2 = new Linear(dModel, dModel)
//
//  def forward(x: Tensor[Float32]): Tensor[Float32] =
//    val out = nn.functional.relu(fc1(x))
//    fc2(out)
//
//// 定义门控网络
//class GatingNetwork(inputDim: Int, numExperts: Int) extends Module:
//  val fc = new Linear(inputDim, numExperts)
//
//  def forward(x: Tensor[Float32]): Tensor[Float32] =
//    nn.functional.softmax(fc(x), dim = 1)
//
//// 定义 MoE 模块
//class MoE(inputDim: Int, dModel: Int, numExperts: Int) extends Module:
//  val experts = ModuleList[Expert](Seq.fill(numExperts)(new torch.Expert(inputDim, dModel)))
//  val gatingNetwork = new GatingNetwork(inputDim, numExperts)
//
//  def forward(x: Tensor[Float32]): Tensor[Float32] =
//    val gates = gatingNetwork(x)
//    val expertOutputs = experts.map(_(x))
//    val stackedOutputs = torch.stack(expertOutputs, dim = 1)
//    torch.sum(gates.unsqueeze(-1) * stackedOutputs, dim = 1)
//
//// 定义 Transformer 块
//class TransformerBlock(inputDim: Int, dModel: Int, numExperts: Int, dFf: Int, dropout: Double) extends Module:
//  val moe = new MoE(inputDim, dModel, numExperts)
//  val norm1 = new LayerNorm(dModel)
//  val dropout1 = new Dropout(dropout)
//
//  val feedForward = nn.Sequential(
//    new Linear(dModel, dFf),
//    new ReLU(),
//    new Linear(dFf, dModel)
//  )
//  val norm2 = new LayerNorm(dModel)
//  val dropout2 = new Dropout(dropout)
//
//  def forward(x: Tensor[Float32]): Tensor[Float32] =
//    val moeOutput = moe(x)
//    var out = norm1(x + dropout1(moeOutput))
//    val ffOutput = feedForward(out)
//    norm2(out + dropout2(ffOutput))
//
//// 定义完整的 MoE Transformer 分类模型
//class MoETransformerClassifier(inputDim: Int, dModel: Int, numExperts: Int, dFf: Int, numLayers: Int, numClasses: Int, dropout: Double) extends Module:
//  val embedding = new Linear(inputDim, dModel)
//  val transformerBlocks = ModuleList[TransformerBlock](
//    Seq.fill(numLayers)(new torch.TransformerBlock(dModel, dModel, numExperts, dFf, dropout))
//  )
//  val fc = new Linear(dModel, numClasses)
//  val dropoutLayer = new Dropout(dropout)
//
//  def forward(x: Tensor[Float32]): Tensor[Float32] =
//    val flattenedX = x.view(x.size(0), -1)
//    var out = embedding(flattenedX)
//    for block <- transformerBlocks do
//      out = block(out)
//    fc(out)
//
//// 模型参数
//val inputDim = 28 * 28
//val dModel = 128
//val numExperts = 4
//val dFf = 512
//val numLayers = 2
//val numClasses = 10
//val dropout = 0.1
//
//// 初始化模型、损失函数和优化器
//val model = new MoETransformerClassifier(inputDim, dModel, numExperts, dFf, numLayers, numClasses, dropout).to(device)
//
//val criterion = new CrossEntropyLoss()
//val optimizer = new Adam(model.parameters(), lr = 0.001)
//
//// 训练模型
//val numEpochs = 10
//for epoch <- 1 to numEpochs do
//  model.train()
//  var totalLoss = 0.0
//  for (images, labels) <- trainLoader do
//    val imagesDevice = images.to(device)
//    val labelsDevice = labels.to(device)
//
//    optimizer.zeroGrad()
//    val outputs = model(imagesDevice).asInstanceOf[Tensor[Float32]]
//    val loss = criterion(outputs, labelsDevice)
//    loss.backward()
//    optimizer.step()
//
//    totalLoss += loss.item()
//
//  val avgLoss = totalLoss / trainLoader.length
//  println(f"Epoch $epoch/$numEpochs, Loss: $avgLoss%.4f")
//
//println("Training finished.")
//
//// 测试模型
//model.eval()
//var correct = 0
//var total = 0
//torch.no_grad {
//  for (images, labels) <- testLoader do
//    val imagesDevice = images.to(device)
//    val labelsDevice = labels.to(device)
//    val outputs = model(imagesDevice).asInstanceOf[Tensor[Float32]]
//    val (_, predicted) = torch.max(outputs.data, 1)
//    total += labels.size(0)
//    correct += (predicted === labelsDevice).sum().item()
//}
//
//println(f"Test Accuracy: ${100.0 * correct / total}%.2f%%")
