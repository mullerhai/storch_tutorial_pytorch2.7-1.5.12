
import scala.collection.mutable.ArrayBuffer
import torch.*
import torch.nn.*
import torch.optim.*
import torch.utils.data.*

// 检查 GPU 是否可用
val device = if torch.cuda.isAvailable() then torch.device("cuda") else torch.device("cpu")
println(s"Using device: $device")

// 数据路径
val DATA_DIR = "./data/criteo_small"
val TRAIN_PATH = new File(DATA_DIR, "train.txt").getPath
val TEST_PATH = new File(DATA_DIR, "test.txt").getPath
val VALS_PATH = new File(DATA_DIR, "val.txt").getPath

// 检查文件是否存在
val filePaths = Seq(TRAIN_PATH, TEST_PATH, VALS_PATH)
if !filePaths.forall(path => new File(path).exists()) then
  throw new FileNotFoundException("请检查数据文件是否存在于指定目录。")

// 读取数据
def read_data(filePath: String): DataFrame =
  DataFrame.read_csv(filePath, sep = "\t", header = None)

val train_df = read_data(TRAIN_PATH)
val test_df = read_data(TEST_PATH)
val vals_df = read_data(VALS_PATH)

// 数据预处理
// 第 0 列是标签，1 - 13 列是数值特征，14 - 39 列是类别特征
val label_col = 0
val numerical_cols = (1 until 14).toList
val categorical_cols = (14 until 40).toList

// 合并数据集用于统一编码
val all_df = pd.concat(Seq(train_df, test_df, vals_df), ignore_index = true)

// 处理缺失值
all_df(categorical_cols) = all_df(categorical_cols).fillna("nan")
all_df(numerical_cols) = all_df(numerical_cols).fillna(0)

// 对类别特征进行编码
val label_encoders = collection.mutable.Map[Int, LabelEncoder]()
for col <- categorical_cols do
  val le = new LabelEncoder()
  all_df(col) = le.fit_transform(all_df(col).astype[String])
  label_encoders(col) = le

// 拆分回训练集、测试集和验证集
val train_df_split = all_df.slice(0, train_df.length)
val test_df_split = all_df.slice(train_df.length, train_df.length + test_df.length)
val vals_df_split = all_df.slice(train_df.length + test_df.length, all_df.length)

// 划分特征和标签
def split_features_labels(df: DataFrame): (Tensor[Float32], Tensor[Float32]) =
  val X = df.drop(columns = Seq(label_col)).values.toTensor[Float32]
  val y = df(label_col).values.toTensor[Float32]
  (X, y)

val (X_train, y_train) = split_features_labels(train_df_split)
val (X_test, y_test) = split_features_labels(test_df_split)
val (X_val, y_val) = split_features_labels(vals_df_split)

// 定义数据集类
class CriteoDataset(X: Tensor[Float32], y: Tensor[Float32]) extends Dataset[Tensor[Float32], Tensor[Float32]]:
  def len(): Int = X.shape(0)
  def apply(idx: Int): (Tensor[Float32], Tensor[Float32]) =
    (X(idx), y(idx))

// 定义 MoE 层
class MoE(num_experts: Int, d_model: Int, num_classes: Int) extends Module:
  val experts = ModuleList[Linear](Seq.fill(num_experts)(new Linear(d_model, num_classes)))
  val gate = new Linear(d_model, num_experts)

  def forward(x: Tensor[Float32]): Tensor[Float32] =
    val gate_output = gate(x)
    val gate_weights = torch.softmax(gate_output, dim = 1)
    val expert_outputs = torch.stack(experts.map(_(x)), dim = 1)
    val output = torch.sum(gate_weights.unsqueeze(-1) * expert_outputs, dim = 1)
    output

// 定义 Transformer MoE 推荐模型
class PositionalEncoding(d_model: Int, max_len: Int = 5000) extends Module:
  val pe = torch.zeros(max_len, d_model)
  val position = torch.arange(0, max_len, dtype = torch.float32).unsqueeze(1)
  val div_term = torch.exp(torch.arange(0, d_model, 2).float32() * (-torch.log(torch.tensor(10000.0))) / d_model)
  pe(::, 0::2) = torch.sin(position * div_term)
  pe(::, 1::2) = torch.cos(position * div_term)
  register_buffer("pe", pe.unsqueeze(0))

  def forward(x: Tensor[Float32]): Tensor[Float32] =
    val pe_slice = pe(::, :x.size(1), ::)
    x + pe_slice

class TransformerMoERecommender(input_dim: Int, d_model: Int, nhead: Int, num_layers: Int, num_experts: Int, num_classes: Int, dropout: Double) extends Module:
  val embedding = new Linear(input_dim, d_model)
  val positional_encoding = new PositionalEncoding(d_model)
  val transformer_encoder = new TransformerEncoder(
    new TransformerEncoderLayer(d_model = d_model, nhead = nhead, dropout = dropout, batch_first = true),
    num_layers = num_layers
  )
  val moe = new MoE(num_experts, d_model, num_classes)
  val dropout_layer = new Dropout(dropout)

  def forward(x: Tensor[Float32]): Tensor[Float32] =
    var x_transformed = embedding(x)
    x_transformed = x_transformed.unsqueeze(1)
    x_transformed = positional_encoding(x_transformed)
    x_transformed = transformer_encoder(x_transformed)
    x_transformed = x_transformed.squeeze(1)
    x_transformed = moe(x_transformed)
    torch.sigmoid(x_transformed).squeeze()

// 初始化数据集和数据加载器
val train_dataset = new CriteoDataset(X_train, y_train)
val test_dataset = new CriteoDataset(X_test, y_test)
val val_dataset = new CriteoDataset(X_val, y_val)

val train_loader = DataLoader(train_dataset, batch_size = 64, shuffle = true)
val test_loader = DataLoader(test_dataset, batch_size = 64, shuffle = false)
val val_loader = DataLoader(val_dataset, batch_size = 64, shuffle = false)

// 模型参数
val INPUT_DIM = X_train.shape(1)
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
  var total_loss = 0.0
  for (X, y) <- dataloader do
    val X_device = X.to(device)
    val y_device = y.to(device)
    optimizer.zero_grad()
    val outputs = model(X_device).asInstanceOf[Tensor[Float32]]
    val loss = criterion(outputs, y_device)
    loss.backward()
    optimizer.step()
    total_loss += loss.item()
  total_loss / dataloader.length

// 评估函数
def evaluate(model: Module, dataloader: DataLoader[Tensor[Float32], Tensor[Float32]], criterion: Loss[Float32]): Double =
  model.eval()
  var total_loss = 0.0
  torch.no_grad {
    for (X, y) <- dataloader do
      val X_device = X.to(device)
      val y_device = y.to(device)
      val outputs = model(X_device).asInstanceOf[Tensor[Float32]]
      val loss = criterion(outputs, y_device)
      total_loss += loss.item()
  }
  total_loss / dataloader.length

// 训练循环
val N_EPOCHS = 10
var best_val_loss = Double.PositiveInfinity
for epoch <- 1 to N_EPOCHS do
  val train_loss = train(model, train_loader, optimizer, criterion)
  val val_loss = evaluate(model, val_loader, criterion)
  println(f"Epoch $epoch/$N_EPOCHS, Train Loss: $train_loss%.4f, Val Loss: $val_loss%.4f")

  if val_loss < best_val_loss then
    best_val_loss = val_loss
    torch.save(model.state_dict(), "best_model.pth")

val test_loss = evaluate(model, test_loader, criterion)
println(f"Test Loss: $test_loss%.4f")
println("Training finished.")
