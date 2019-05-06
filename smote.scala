import org.apache.spark.ml.feature.BucketedRandomProjectionLSH
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.sql.expressions.Window
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.sql.functions.rand
import org.apache.spark.sql.functions._

object smoteClass{
  //object with three methods
  def KNNCalculation(
    dataFinal:org.apache.spark.sql.DataFrame,
    features:String,
    reqrows:Int,
    BucketLength:Int,
    NumHashTables:Int):org.apache.spark.sql.DataFrame = {
      val b1 = dataFinal.withColumn("index", row_number().over(Window.partitionBy(features).orderBy(features)))
      val brp = new BucketedRandomProjectionLSH().setBucketLength(BucketLength).setNumHashTables(NumHashTables).setInputCol(features).setOutputCol("values")
      val model = brp.fit(b1)
      //so you essentially approx join two transformed dfs and calc the distance
      val transformedA = model.transform(b1)
      val transformedB = model.transform(b1)
      val b2 = model.approxSimilarityJoin(transformedA, transformedB, 2000000000.0)
      require(b2.count > reqrows, println("Change bucket length or reduce the percentageOver"))
      //you select the columns you want , filter for where there is a distance, order the df by dist
      //drop duplicates and limit it to the req amount
      val b3 = b2.selectExpr("datasetA.index as id1",
        "datasetA.features as k1",
        "datasetB.index as id2",
        "datasetB.features as k2",
        "distCol").filter("distCol>0.0").orderBy("id1", "distCol").dropDuplicates().limit(reqrows)
      return b3
  }

  def smoteCalc(key1: org.apache.spark.ml.linalg.Vector, key2: org.apache.spark.ml.linalg.Vector)={
    val resArray = Array(key1, key2)
    //this looks really cool, but I need more context. You take two vectors and then you zip them together
    //this creates a tuple. this tuple you first sum together and then multiply by 0,2
    //After which you create another tuple again with the first vector and sum them again
    //this you then concanate with the original array of two vectors as a dense vector resulting in an array of three vectors 
    val res = key1.toArray.zip(key2.toArray.zip(key1.toArray).map(x => x._1 - x._2).map(_*0.2))
    .map(x => x._1 + x._2)
    resArray :+ org.apache.spark.ml.linalg.Vectors.dense(res)}

  def Smote(
    inputFrame:org.apache.spark.sql.DataFrame,
    features:String,
    label:String,
    oversampleRate:Int,
    BucketLength:Int,
    NumHashTables:Int):org.apache.spark.sql.DataFrame = {
      //you simply select and calc the amount of rows required for the oversample rate
      val frame = inputFrame.select(features,label).where(label + " == " + 1)
      val rowCount = frame.count
      val reqrows = (rowCount * oversampleRate).toInt
      //udf function of smotecalc method
      val md = udf(smoteCalc _)
      //calling the knncalc method, this results in a dataframe with ids, features and distcol
      val b1 = KNNCalculation(frame, features, reqrows, BucketLength, NumHashTables)
      //using the smote udf you use it on the two feature cols and then select it
      val b2 = b1.withColumn("ndtata", md($"k1", $"k2")).select("ndtata")
      //you explode the array of three vectores and dropduplicates
      val b3 = b2.withColumn("AllFeatures", explode($"ndtata")).select("AllFeatures").dropDuplicates
      //you add label column and correct schema
      val b4 = b3.withColumn(label, lit(1).cast(frame.schema(1).dataType))
      return b4
  }
}
