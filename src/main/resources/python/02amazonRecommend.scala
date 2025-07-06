
import scala.collection.mutable.{ArrayBuffer, DefaultDict}
import torch.*
import torch.nn.*
import torch.optim.*
import torch.utils.data.*

// 检查 GPU 是否可用
val device = if torch.cuda.isAvailable() then torch.device("cuda") else torch.device("cpu")
println(s"Using device: $device")

// 下载并加载 Amazon 数据集（以 Electronics 类别为例）
val DATA_URL = "http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Electronics_5.json.gz"
val DATA_PATH = "reviews_Electronics_5.json.gz"
println("Downloading and loading Amazon dataset...")

if !Files.exists(Paths.get(DATA_PATH)) then
  println("Downloading Amazon dataset...")
  val url = new URL(DATA_URL)
  val in = url.openStream()
  Files.copy(in, Paths.get(DATA_PATH))
  in.close()

// 解析数据
val user_item_dict = DefaultDict[Int, ArrayBuffer[Int]](() => ArrayBuffer.empty[Int])
val item_user_dict = DefaultDict[Int, ArrayBuffer[Int]](() => ArrayBuffer.empty[Int])
val user_id_map = collection.mutable.Map[String, Int]()
val item_id_map = collection.mutable.Map[String, Int]()
var user_counter = 0
var item_counter = 0

println("Loading Amazon dataset ungzip ...")
val gzFile = new java.util.zip.GZIPInputStream(new java.io.FileInputStream(DATA_PATH))
val reader = new java.io.BufferedReader(new java.io.InputStreamReader(gzFile, "UTF-8"))
var line = reader.readLine()
while line != null do
  val data = ujson.read(line)
  val user_id = data("reviewerID").str
  val item_id = data("asin").str

  if !user_id_map.contains(user_id) then
    user_id_map(user_id) = user_counter
    user_counter += 1
  if !item_id_map.contains(item_id) then
    item_id_map(item_id) = item_counter
    item_counter += 1

  val user_idx = user_id_map(user_id)
  val item_idx = item_id_map(item_id)
  user_item_dict(user_idx).append(item_idx)
  item_user_dict(item_idx).append(user_idx)

  line = reader.readLine()
reader.close()

// 生成训练和测试数据
val all_interactions = ArrayBuffer[(Int, Int, Int)]()
for user <- user_item_dict.keys do
  for item <- user_item_dict(user) do
    all_interactions.append((user, item, 1)) // 1 表示有交互

println("Generating training and testing data...")
// 生成负样本
val num_negatives = 4
var index = 0
for user <- user_item_dict.keys do
  index += 1
  val all_items = (0 until item_counter).toSet
  val interacted_items = user_item_dict(user).toSet
  val non_interacted_items = all_items -- interacted_items
  val sampled_negatives = np.random.choice(non_interacted_items.toArray, num_negatives * interacted_items.size).toArray
  println(s"user_id: $user, index: $index")
  for item <- sampled_negatives do
    all_interactions.append((user, item, 0)) // 0 表示无交互
println("Training and testing data generated.")

val (train_interactions, test_interactions) = train_test_split(all_interactions.toArray, test_size = 0.2, random_state = 42)

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
class TransformerMoERecommender(num_users: Int, num_items: Int, d_model: Int, nhead: Int, num_layers: Int, num_experts: Int, dropout: Double) extends Module:
  val user_embedding = new Embedding(num_users, d_model)
  val item_embedding = new Embedding(num_items, d_model)
  val transformer_encoder = new TransformerEncoder(
    new TransformerEncoderLayer(d_model = d_model, nhead = nhead, dropout = dropout),
    num_layers = num_layers
  )
  val moe = new MoE(num_experts, d_model, 1)
  val dropout_layer = new Dropout(dropout)

  def forward(user_ids: Tensor[Long], item_ids: Tensor[Long]): Tensor[Float32] =
    val user_embed = user_embedding(user_ids).asInstanceOf[Tensor[Float32]]
    val item_embed = item_embedding(item_ids).asInstanceOf[Tensor[Float32]]
    val combined_embed = user_embed + item_embed
    val combined_embed_seq = combined_embed.unsqueeze(0) // 添加序列维度
    var output = transformer_encoder(combined_embed_seq)
    output = output.squeeze(0)
    output = moe(output)
    torch.sigmoid(output).squeeze()

// 定义数据集类
class AmazonDataset(interactions: Array[(Int, Int, Int)]) extends Dataset[Tensor[Long], (Tensor[Long], Tensor[Float32])]:
  def len(): Int = interactions.length
  def apply(idx: Int): (Tensor[Long], Tensor[Long], Tensor[Float32]) =
    val (user, item, label) = interactions(idx)
    (torch.tensor(user, dtype = torch.long), torch.tensor(item, dtype = torch.long), torch.tensor(label, dtype = torch.float32))

println("Initializing Amazon dataset and data loaders...")
// 初始化数据集和数据加载器
val train_dataset = new AmazonDataset(train_interactions)
val test_dataset = new AmazonDataset(test_interactions)
val train_loader = DataLoader(train_dataset, batch_size = 64, shuffle = true)
val test_loader = DataLoader(test_dataset, batch_size = 64, shuffle = false)
println("Amazon dataset and data loaders initialized complete.")

// 模型参数
val NUM_USERS = user_counter
val NUM_ITEMS = item_counter
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
  var total_loss = 0.0
  for (user_ids, item_ids, labels) <- dataloader do
    val user_ids_device = user_ids.to(device)
    val item_ids_device = item_ids.to(device)
    val labels_device = labels.to(device)
    optimizer.zero_grad()
    val outputs = model(user_ids_device, item_ids_device).asInstanceOf[Tensor[Float32]]
    val loss = criterion(outputs, labels_device)
    loss.backward()
    optimizer.step()
    total_loss += loss.item()
    println(s"Batch loss: ${loss.item()}")
  total_loss / dataloader.length

// 评估函数
def evaluate(model: Module, dataloader: DataLoader[Tensor[Long], (Tensor[Long], Tensor[Float32])], criterion: Loss[Float32]): Double =
  model.eval()
  var total_loss = 0.0
  torch.no_grad {
    for (user_ids, item_ids, labels) <- dataloader do
      val user_ids_device = user_ids.to(device)
      val item_ids_device = item_ids.to(device)
      val labels_device = labels.to(device)
      val outputs = model(user_ids_device, item_ids_device).asInstanceOf[Tensor[Float32]]
      val loss = criterion(outputs, labels_device)
      total_loss += loss.item()
      println(s"evaluation Batch loss: ${loss.item()}")
  }
  total_loss / dataloader.length

// 训练循环
val N_EPOCHS = 10
for epoch <- 1 to N_EPOCHS do
  println(s"EpochRange ${epoch}/${N_EPOCHS}")
  val train_loss = train(model, train_loader, optimizer, criterion)
  val test_loss = evaluate(model, test_loader, criterion)
  println(f"Epoch $epoch/$N_EPOCHS, Train Loss: $train_loss%.4f, Test Loss: $test_loss%.4f")

println("Training finished.")
