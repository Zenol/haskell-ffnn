module Main where

import MNIST
import NeuralNetwork
import Math.LinearAlgebra.Sparse.Matrix
import Math.LinearAlgebra.Sparse.Vector
import Data.Word
import qualified Data.Vector as V
import System.Random
import qualified Data.List as L

dtSet :: [(V.Vector Word8, Word8)] -> [(Input Double, Output Double)]
dtSet dtMNIST = fmap sparsify dtMNIST
  where
    sparsify (x, y) = (sparseList . map fromIntegral . V.toList $ x,
                       vecFromAssocList [(1 + fromIntegral y, 1)])

inlineLearning :: (Input Double -> Output Double -> Network Double -> Network Double)
   -> Network Double
   -> [(Input Double, Output Double)]
   -> Network Double
inlineLearning learnFct net dt = L.foldl' (flip . uncurry $ learnFct) net dt

simp x = floor $ x * 10000

argmaxs xs = L.findIndices  (==maximum xs) xs
-- Assert non empty list
argmax xs = let Just v = L.findIndex  (==maximum xs) xs in fromIntegral v

countEquals x y = fromIntegral . length . filter (==True) $ zipWith (==) x y

main = do
  seed <- newStdGen
  let rdList = randomRs (0.0, 1.0) seed
  dt <- loadMNIST "train-labels.idx1-ubyte" "train-images.idx3-ubyte"
  putStrLn "Step size (in Double) ?"
  --step <- (fmap read getLine) :: IO Double
  let step = 3
  mnist <- loadMNIST "train-images.idx3-ubyte" "train-labels.idx1-ubyte"
  let net = inlineLearning (learn step) (networkDigits rdList) (take 5000 $ dtSet mnist)
  -- All the images
  let x = map (sparseList . map fromIntegral . V.toList . fst) (take 3000 mnist)
  let y = map (snd) (take 10000 mnist)
  let x' = map fillVec $ map (flip eval $ net) x
  let x'' = map argmax $ x'
  
  --putStrLn (show . layers $ net)
  putStrLn ""
  putStrLn (show $ take 10 y)
  putStrLn (show $ take 10 x')
  putStrLn (show $ take 10 x'')
  putStrLn "Efficiency : "
  putStrLn (show $ (countEquals x'' y) / (fromIntegral . length $ x) * 100)
  return net
