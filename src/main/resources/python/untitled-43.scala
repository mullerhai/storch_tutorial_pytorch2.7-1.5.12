import java.io.{BufferedReader, InputStreamReader}
import java.net.URL
import java.nio.file.{Files, Paths}
import java.util.zip.GZIPInputStream
import scala.collection.mutable
import scala.util.Random
import org.platanios.tensorflow.api._
import org.platanios.torch.jni
import org.platanios.torch.api._
import org.platanios.torch.api.data.{Dataset, DataLoader}
import org.platanios.torch.api.tensors.Tensor

class AmazonDataset(root: String, train: Boolean = true, download: Boolean = false) extends Dataset[(Tensor, Tensor, Tensor)] {
  private val dataUrl = "http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Electronics_5.json.gz"
  private val dataPath = Paths.get(root, "reviews_Electronics_5.json.gz")

  if (download) {
    downloadData()
  }

  if (!Files.exists(dataPath)) {
    throw new RuntimeException("Dataset not found. You can use download = true to download it.")
  }

  private val userItemDict = mutable.Map[Int, mutable.ListBuffer[Int]]().withDefaultValue(mutable.ListBuffer.empty[Int])
  private val itemUserDict = mutable.Map[Int, mutable.ListBuffer[Int]]().withDefaultValue(mutable.ListBuffer.empty[Int])
  private val userIdMap = mutable.Map[String, Int]()
  private val itemIdMap = mutable.Map[String, Int]()
  private var userCounter = 0
  private var itemCounter = 0
  private var interactions: Seq[(Int, Int, Int)] = _

  loadData()
  generateInteractions()

  private def downloadData(): Unit = {
    if (Files.exists(dataPath)) return
    Files.createDirectories(Paths.get(root))
    println("Downloading Amazon dataset...")
    val in = new URL(dataUrl).openStream()
    val out = Files.newOutputStream(dataPath)
    in.transferTo(out)
    in.close()
    out.close()
    println("Download complete.")
  }

  private def loadData(): Unit = {
    println("Loading Amazon dataset ungzip ...")
    val gis = new GZIPInputStream(Files.newInputStream(dataPath))
    val reader = new BufferedReader(new InputStreamReader(gis))
    var line = reader.readLine()
    while (line != null) {
      val data = ujson.read(line).obj
      val userId = data("reviewerID").str
      val itemId = data("asin").str

      if (!userIdMap.contains(userId)) {
        userIdMap(userId) = userCounter
        userCounter += 1
      }
      if (!itemIdMap.contains(itemId)) {
        itemIdMap(itemId) = itemCounter
        itemCounter += 1
      }

      val userIdx = userIdMap(userId)
      val itemIdx = itemIdMap(itemId)
      userItemDict(userIdx) += itemIdx
      itemUserDict(itemIdx) += userIdx

      line = reader.readLine()
    }
    reader.close()
    gis.close()
  }

  private def generateInteractions(): Unit = {
    var allInteractions = mutable.ListBuffer[(Int, Int, Int)]()

    // 生成正样本
    userItemDict.foreach { case (user, items) =>
      items.foreach { item =>
        allInteractions += ((user, item, 1))
      }
    }

    // 生成负样本
    val numNegatives = 4
    userItemDict.foreach { case (user, interactedItems) =>
      val allItems = (0 until itemCounter).toSet
      val nonInteractedItems = allItems -- interactedItems
      val sampledNegatives = Random.shuffle(nonInteractedItems.toList).take(numNegatives * interactedItems.size)
      sampledNegatives.foreach { item =>
        allInteractions += ((user, item, 0))
      }
    }

    // 划分训练集和测试集
    val trainSize = (allInteractions.size * 0.8).toInt
    interactions = if (train) allInteractions.take(trainSize) else allInteractions.drop(trainSize)
  }

  override def size: Int = interactions.size

  override def apply(index: Int): (Tensor, Tensor, Tensor) = {
    val (user, item, label) = interactions(index)
    (
      Tensor(user, dtype = Int64),
      Tensor(item, dtype = Int64),
      Tensor(label.toFloat, dtype = Float32)
    )
  }
}

object AmazonDatasetExample extends App {
  // 初始化训练集
  val trainDataset = new AmazonDataset(root = "./data", train = true, download = true)
  // 初始化测试集
  val testDataset = new AmazonDataset(root = "./data", train = false)

  println(s"Train dataset size: ${trainDataset.size}")
  println(s"Test dataset size: ${testDataset.size}")

  // 获取第一个样本
  val (user, item, label) = trainDataset(0)
  println(s"User: ${user}, Item: ${item}, Label: ${label}")

  // 使用 DataLoader
  val trainLoader = DataLoader(trainDataset, batchSize = 64, shuffle = true)
  trainLoader.foreach { batch =>
    val (users, items, labels) = batch
    println(s"Batch - Users shape: ${users.shape}, Items shape: ${items.shape}, Labels shape: ${labels.shape}")
  }
}
