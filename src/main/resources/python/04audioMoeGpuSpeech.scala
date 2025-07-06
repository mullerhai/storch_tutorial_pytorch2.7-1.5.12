import torch.{GatingNetwork, MoE, *}
import torch.nn.*
import torch.optim.*
import torch.utils.data.*

import scala.collection.mutable.ArrayBuffer

// 检查 GPU 是否可用
val device = if torch.cuda.isAvailable() then torch.device("cuda") else torch.device("cpu")
println(s"Using device: $device")

// 下载并加载 LIBRISPEECH 数据集
val trainDataset = LibriSpeech(
  root = Paths.get("data"),
  url = "train-clean-100",
  download = true
)
val testDataset = LibriSpeech(
  root = Paths.get("data"),
  url = "test-clean",
  download = true
)

// 定义字符映射
val charMapStr =
  """
    |' 0
    |<SPACE> 1
    |a 2
    |b 3
    |c 4
    |d 5
    |e 6
    |f 7
    |g 8
    |h 9
    |i 10
    |j 11
    |k 12
    |l 13
    |m 14
    |n 15
    |o 16
    |p 17
    |q 18
    |r 19
    |s 20
    |t 21
    |u 22
    |v 23
    |w 24
    |x 25
    |y 26
    |z 27
    |""".stripMargin

val charMap = collection.mutable.Map[String, Int]()
val indexMap = collection.mutable.Map[Int, String]()
for line <- charMapStr.strip.split("\n") if line.nonEmpty do
  val Array(ch, index) = line.split(" ")
  charMap(ch) = index.toInt
  indexMap(index.toInt) = ch
indexMap(1) = " "

// 数据预处理函数
def dataProcessing(data: Seq[(Tensor[Float32], Int, String, String, String, String)], charMap: collection.mutable.Map[String, Int]): (Tensor[Float32], Tensor[Long], Seq[Int], Seq[Int]) =
  val spectrograms = ArrayBuffer[Tensor[Float32]]()
  val labels = ArrayBuffer[Tensor[Long]]()
  val inputLengths = ArrayBuffer[Int]()
  val labelLengths = ArrayBuffer[Int]()

  for (waveform, _, utterance, _, _, _) <- data do
    val spectrogram = torchaudio.transforms.MelSpectrogram(
      sampleRate = 16000,
      nMels = 128
    ).apply(waveform).squeeze(0).transpose(0, 1)
    spectrograms.append(spectrogram)

    val label = torch.tensor(
      utterance.toLowerCase.replace(" ", "<SPACE>").map(c => charMap.getOrElse(c.toString, charMap("<SPACE>"))).toArray,
      dtype = torch.long
    )
    labels.append(label)

    inputLengths.append(spectrogram.shape(0) / 2)
    labelLengths.append(label.size(0))

  val paddedSpectrograms = torch.nn.utils.rnn.pad_sequence(spectrograms.toSeq, batchFirst = true).unsqueeze(1).transpose(2, 3)
  val paddedLabels = torch.nn.utils.rnn.pad_sequence(labels.toSeq, batchFirst = true)

  (paddedSpectrograms, paddedLabels, inputLengths.toSeq, labelLengths.toSeq)

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

// 定义语音识别模型
class SpeechRecognitionModel(inputDim: Int, dModel: Int, numExperts: Int, dFf: Int, nhead: Int, numLayers: Int, numClasses: Int, dropout: Double) extends Module:
  val embedding = new Linear(inputDim, dModel)
  val transformerBlocks = ModuleList[TransformerBlock](
    Seq.fill(numLayers)(new TransformerBlock(dModel, dModel, numExperts, dFf, nhead, dropout))
  )
  val fc = new Linear(dModel, numClasses)

  def forward(x: Tensor[Float32]): Tensor[Float32] =
    var out = x.squeeze(1).transpose(1, 2)
    out = embedding(out)
    for block <- transformerBlocks do
      out = block(out)
    out = fc(out)
    out.transpose(0, 1)

// 模型参数
val inputDim = 128
val dModel = 256
val numExperts = 4
val dFf = 1024
val nhead = 4
val numLayers = 2
val numClasses = charMap.size
val dropout = 0.1

// 初始化模型、损失函数和优化器
val model = new SpeechRecognitionModel(
  inputDim, dModel, numExperts, dFf, nhead, numLayers, numClasses, dropout
).to(device)
val criterion = new CTCLoss(blank = 0)
val optimizer = new Adam(model.parameters(), lr = 1e-4)

// 训练函数
def train(model: Module, device: Device, trainLoader: DataLoader[(Tensor[Float32], Tensor[Long], Seq[Int], Seq[Int])], criterion: Loss[Float32], optimizer: Optimizer, epoch: Int): Unit =
  model.train()
  val dataLen = trainLoader.dataset.len()
  for (batchIdx, batch) <- trainLoader.zipWithIndex do
    val (spectrograms, labels, inputLengths, labelLengths) = batch
    val spectrogramsDevice = spectrograms.to(device)
    val labelsDevice = labels.to(device)

    optimizer.zero_grad()

    val output = model(spectrogramsDevice).asInstanceOf[Tensor[Float32]]
    val logSoftmaxOutput = nn.functional.log_softmax(output, dim = 2)
    val loss = criterion(logSoftmaxOutput, labelsDevice, inputLengths, labelLengths)
    loss.backward()

    optimizer.step()
    if batchIdx % 100 == 0 || batchIdx == dataLen then
      println(f"Train Epoch: $epoch [${batchIdx * spectrograms.size(0)}/$dataLen (${100.0 * batchIdx / trainLoader.len()}%.0f%%)]\tLoss: ${loss.item()}%.6f")

// 测试函数
def test(model: Module, device: Device, testLoader: DataLoader[(Tensor[Float32], Tensor[Long], Seq[Int], Seq[Int])], criterion: Loss[Float32]): Unit =
  model.eval()
  var testLoss = 0.0
  torch.no_grad {
    for batch <- testLoader do
      val (spectrograms, labels, inputLengths, labelLengths) = batch
      val spectrogramsDevice = spectrograms.to(device)
      val labelsDevice = labels.to(device)

      val output = model(spectrogramsDevice).asInstanceOf[Tensor[Float32]]
      val logSoftmaxOutput = nn.functional.log_softmax(output, dim = 2)
      val loss = criterion(logSoftmaxOutput, labelsDevice, inputLengths, labelLengths)
      testLoss += loss.item()
  }
  testLoss /= testLoader.dataset.len()
  println(f"Test set: Average loss: $testLoss%.4f\n")

// 数据加载器
val trainLoader = DataLoader(
  dataset = trainDataset,
  batchSize = 20,
  shuffle = true,
  collateFn = (data: Seq[(Tensor[Float32], Int, String, String, String, String)]) => dataProcessing(data, charMap)
)
val testLoader = DataLoader(
  dataset = testDataset,
  batchSize = 20,
  shuffle = false,
  collateFn = (data: Seq[(Tensor[Float32], Int, String, String, String, String)]) => dataProcessing(data, charMap)
)

// 训练和测试
val numEpochs = 10
for epoch <- 1 to numEpochs do
  train(model, device, trainLoader, criterion, optimizer, epoch)
  test(model, device, testLoader, criterion)
