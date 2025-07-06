package torch

import torch.nn.*
import torch.nn
import torch.{Float32, Tensor}
import torch.nn.functional as F
import torch.nn.modules.TensorModule
import torch.Device.{CPU, CUDA}
import torch.optim.Adam

object  mn extends App{

//  @main
  def main(): Unit = {
    val inputDim = 28 * 28
    val dModel = 128
    val numExperts = 4
    val dFf = 512
    val numLayers = 2
    val numClasses = 10
    val dropout = 0.1
    val device = if torch.cuda.isAvailable then CUDA else CPU
    println(s"Using device: $device")
    // 初始化模型、损失函数和优化器
    val model = new MoETransformerClassifier[Float32](inputDim, dModel, numExperts, dFf, numLayers, numClasses, dropout).to(device)

    val criterion = new CrossEntropyLoss()
    val optimizer = new Adam(model.parameters(true), lr = 0.001)
    val batchRandTensor = torch.randn(32, 28, 28)(dtype = torch.float32, requires_grad = true)
    val output = model(batchRandTensor)
    println(output.shape.mkString(","))

  }


}

class Expert[ParamType <: FloatNN | ComplexNN: Default](inputDim: Int, dModel: Int) extends TensorModule[ParamType]:

  val fc1 = register(nn.Linear(inputDim, dModel))
  val fc2 = register(nn.Linear(dModel, dModel))

  def apply(x: Tensor[ParamType]): Tensor[ParamType] =
    forward(x).to(CUDA)

  def forward(x: Tensor[ParamType]): Tensor[ParamType] =
    val fa = fc1(x)
    val out = F.relu(fc1(x)).to(CUDA)
    fc2(out)

// 定义门控网络
class GatingNetwork[ParamType <: FloatNN | ComplexNN: Default](inputDim: Int, numExperts: Int) extends TensorModule[ParamType]:

  val fc = register(nn.Linear(inputDim, numExperts)).to(CUDA)

  def forward(x: Tensor[ParamType]): Tensor[ParamType] =
    F.softmax(fc(x), dim = 1).to(CUDA)

  def apply(x: Tensor[ParamType]): Tensor[ParamType] =
    forward(x).to(CUDA)

// 定义 MoE 模块
class MoE[ParamType <: FloatNN | ComplexNN: Default](inputDim: Int, dModel: Int, numExperts: Int) extends TensorModule[ParamType]:

  val expertSeq = Seq.fill(numExperts)(new Expert[ParamType](inputDim, dModel))
  val experts = ModuleList[ParamType](expertSeq*).to(CUDA)
  val gatingNetwork = new GatingNetwork[ParamType](inputDim, numExperts).to(CUDA)

  def apply(x: Tensor[ParamType]): Tensor[ParamType] =
    forward(x).to(CUDA)

  def forward(x: Tensor[ParamType]): Tensor[ParamType] =
    val gates = gatingNetwork(x)
    val expertOutputs = experts.map(exp =>exp(x))
    val stackedOutputs = torch.stack[ParamType](expertOutputs.toSeq, dim = 1)
    val gatesSquash = gates.unsqueeze(-1)
    val gateStack = gatesSquash * stackedOutputs
    torch.sum(gateStack)

// 定义 Transformer 块
class TransformerBlock[ParamType <: FloatNN | ComplexNN: Default](inputDim: Int, dModel: Int, numExperts: Int, dFf: Int, dropout: Double) extends TensorModule[ParamType]:

  val moe = new MoE[ParamType](inputDim, dModel, numExperts).to(CUDA)
  val norm1 = register(nn.LayerNorm(Seq(dModel))).to(CUDA)
  val dropout1 = register(nn.Dropout(dropout.toFloat)).to(CUDA)

  val feedForward = nn.Sequential(
    nn.Linear(dModel, dFf),
    nn.ReLU(),
    nn.Linear(dFf, dModel)
  ).to(CUDA)
  val norm2 = register(nn.LayerNorm(Seq(dModel))).to(CUDA)
  val dropout2 = register(nn.Dropout(dropout.toFloat)).to(CUDA)

  def apply(x: Tensor[ParamType]): Tensor[ParamType] =
    forward(x).to(CUDA)

  def forward(x: Tensor[ParamType]): Tensor[ParamType] =
    val moeOutput = moe(x).to(CUDA)
    var out = norm1(x + dropout1(moeOutput))
    val ffOutput = feedForward(out)
    norm2(out + dropout2(ffOutput))

// 定义完整的 MoE Transformer 分类模型
class MoETransformerClassifier[ParamType <: FloatNN | ComplexNN: Default](inputDim: Int, dModel: Int, numExperts: Int, dFf: Int, numLayers: Int, numClasses: Int, dropout: Double) extends TensorModule[ParamType]:

  val embedding = register(nn.Linear(inputDim, dModel))
  val transformerBlocks = ModuleList(
    Seq.fill(numLayers)(new TransformerBlock[ParamType](dModel, dModel, numExperts, dFf, dropout))*
  ).to(CUDA)
  val fc = register(nn.Linear(dModel, numClasses))
  val dropoutLayer = register(nn.Dropout(dropout.toFloat))

  def apply(x: Tensor[ParamType]): Tensor[ParamType] =
    forward(x).to(CUDA)

  def forward(x: Tensor[ParamType]): Tensor[ParamType] =
    val flattenedX = x.view(x.size(0), -1)
    var out = embedding(flattenedX)
    for block <- transformerBlocks do
      out = block(out)
    fc(out)

