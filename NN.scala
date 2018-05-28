import breeze.linalg._
import breeze.math._
import breeze.numerics._

//need to change from output one to the same amount of output neurons
//calculate error with input
//rank examples per amount of error?
//Should re-read paper

def sigmoid_prime(x: DenseMatrix[Double]) : DenseMatrix[Double] = {
  var f = sigmoid(x)
  var x_row =x.rows
  var x_col =x.cols
  var oneMatrix = DenseMatrix.tabulate(x_row, x_col){case i => 1.0}
  var df =f *:* (oneMatrix - f)
  return df
}
  
var LR= 0.1
val input_size = 4
val hidden_size = 4
val output_size = 1
val epochs = 5000

def initialize_weights(input: Int, hidden: Int, output: Int) :  Tuple2[DenseMatrix[Double], DenseMatrix[Double]] = {
//Generate random weights of appropriote size
  var w1 = DenseMatrix.rand(hidden, input)
  var w2 = DenseMatrix.rand(output, hidden)
  return (w1, w2)
}

var (w1 : DenseMatrix[Double], w2 : DenseMatrix[Double]) = initialize_weights(input_size, hidden_size, output_size)

var X = DenseMatrix((0.0, 1.0, 1.0) , (0.0, 1.0, 1.0), (1.0, 1.0, 1.0), (1.0, 1.0, 1.0) )
var Y = DenseMatrix(0.0, 1.0, 1.0).t

val m = X.cols

//function computes forward
def forward(X :DenseMatrix[Double], w1 : DenseMatrix[Double], w2 : DenseMatrix[Double]) : Tuple4[DenseMatrix[Double],DenseMatrix[Double],DenseMatrix[Double],DenseMatrix[Double]] = {
  var z1 = w1 * X
  var a1 = sigmoid(z1)
  var z2 = w2 * a1
  var a2 = sigmoid(z2)
  
  return (z1, a1, z2, a2)
}
//var (z1, a1, z2, a2) = forward(X, w1, w2)

//non-function implemntation
for (i <- 1 to epochs) 
{
  var z1 = w1 * X

  var a1 = sigmoid(z1)
  var z2 = w2 * a1 //a1 * w2
  var a2 = sigmoid(z2)
    
  var dZ2 = a2 - Y
    
  //backward
  var dZ1 = (w2.t * dZ2) *:* sigmoid_prime(z1) //(dZ2 * w2.t) *:* sigmoid_prime(z1)
  
  var dw1 = (1.0/m) *:* (dZ1 * X.t)
  
  w1 -= LR *:* dw1
  
  var dw2 = (1.0/m) *:* (dZ2 * a1.t)
  
  w2 -= LR * dw2
}

//you have the W2
//you feedforward
var (z1, a1, z2, a2) = forward(X, w1, w2)
//given that the X is the Y, you now calculate the error per m (I thnk that is col in our case)
// and then create a cutoff rate for the error as outlier detection



//functions require too much inputs hmm, not sure if the gain of the functions outweight the extra characters
//change names to free up scope
/***
//function computes forward
def forward(X :DenseMatrix[Double], w1 : DenseMatrix[Double], w2 : DenseMatrix[Double]) : Tuple4[DenseMatrix[Double],DenseMatrix[Double],DenseMatrix[Double],DenseMatrix[Double]] = {
  var z1 = w1 * X
  var a1 = sigmoid(z1)
  var z2 = w2 * a1
  var a2 = sigmoid(z2)
  
  return (z1, a1, z2, a2)
}
//var (z1, a1, z2, a2) = forward(X, w1, w2)

//function computes backpropagates and updates weights
def backpropUpdate (m: Int ,LR: Double ,X :DenseMatrix[Double], w1 : DenseMatrix[Double], w2 : DenseMatrix[Double], z1 :DenseMatrix[Double], a1 : DenseMatrix[Double], z2 : DenseMatrix[Double], a2 :DenseMatrix[Double], Y : DenseMatrix[Double]) : Tuple2[DenseMatrix[Double], DenseMatrix[Double]] = {
  var dZ1 = (w2.t * dZ2) *:* sigmoid_prime(z1) //(dZ2 * w2.t) *:* sigmoid_prime(z1)
  var dw1 = (1.0/m) *:* (dZ1 * X.t)
  w1 -= LR *:* dw1
  var dw2 = (1.0/m) *:* (dZ2 * a1.t)
  w2 -= LR * dw2
  
  return (w1, w2)
}

//function does iterations of the previous functions
def SGD (epochs : Int, m : Int, LR : Double, X:DenseMatrix[Double], Y: DenseMatrix[Double], w1: DenseMatrix[Double], w2: DenseMatrix[Double]) : Tuple2[DenseMatrix[Double], DenseMatrix[Double]] = {
  for (i <- 1 to epochs) {
   
    var (z1, a1, z2, a2) = forward(X, w1 : DenseMatrix[Double], w2: DenseMatrix[Double])
    var (w1 : DenseMatrix[Double], w2: DenseMatrix[Double]) = backpropUpdate(m, LR ,X, w1, w2, z1, a1, z2, a2, Y)

  }
  return (w1, w2)
}

var (trainw1, trainw2) = SGD(epochs, m, LR, X, Y, w1, w2)
***/
