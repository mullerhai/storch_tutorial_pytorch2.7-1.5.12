package torch

// ... 已有代码 ...

import org.bytedeco.pytorch.{ChunkRandomDataLoader, Example, ExampleIterator}
import scala.collection.Iterator

// 定义一个可迭代的类，用于遍历 ChunkRandomDataLoader
class DataLoaderIterable(dataLoader: ChunkRandomDataLoader) extends Iterable[Example] {
  override def iterator: Iterator[Example] = new Iterator[Example] {
    private var current: ExampleIterator = dataLoader.begin
    private val endIterator: ExampleIterator = dataLoader.end

    // 检查是否还有下一个元素
    override def hasNext: Boolean = !current.equals(endIterator)

    // 获取下一个元素并移动迭代器
    override def next(): Example = {
      val batch = current.access
      current = current.increment
      batch
    }
  }
}

object mnistMoeTraining extends App {

  @main
  def main(): Unit =
    // ... 已有代码 ...

    val ds = new ChunkSharedBatchDataset(
      new ChunkDataset(
        reader,
        new RandomSampler(exampleSeq.size),
        new RandomSampler(exampleSeq.size),
        new ChunkDatasetOptions(prefetch_count, batch_size)
      )
    ).map(new ExampleStack)

    val opts = new DataLoaderOptions(100)
    opts.batch_size.put(100)
    val data_loader = new ChunkRandomDataLoader(ds, opts)

    // 使用 DataLoaderIterable 进行遍历
    val iterableDataLoader = new DataLoaderIterable(data_loader)
    (1 to num_epochs).foreach(epoch => {
      var batchIndex = 0
      println("coming in for loop")
      for (batch <- iterableDataLoader) {
        Using.resource(new PointerScope()) { p =>
          optimizer.zeroGrad()
          val prediction = model(fromNative(batch.data().view(-1, 28 * 28)).reshape(-1, 28 * 28).to(device))
          val loss = criterion(prediction, fromNative(batch.target()).to(device))
          loss.backward()
          optimizer.step()
          batchIndex += 1
          if batchIndex % 200 == 0 then
            torch.noGrad {
              println("coming eval...")
              val predictions = model(evalFeatures.reshape(-1, 28 * 28))
              println(s"predictions : ${predictions} \n")
              val evalLoss = criterion(predictions, evalTargets)
              println(s"evalLoss : ${evalLoss.item} \n")
              val accuracy =
                (predictions.argmax(dim = 1).eq(evalTargets).sum / test_dataset.length).item
              println(
                f"Epoch: $epoch | Batch: $batchIndex%4d | Training loss: ${loss.item}%.4f | Eval loss: ${evalLoss.item}%.4f | Eval accuracy: $accuracy%.4f"
              )
            }
        }
      }
    })

    println(s"model ${model.modules.toSeq.mkString(" \n")}")
    println(s"model ${model.summarize}")
}
