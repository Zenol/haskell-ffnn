module NeuralNetwork where

import           Math.LinearAlgebra.Sparse.Matrix
import           Math.LinearAlgebra.Sparse.Vector
import qualified Data.Vector as V


data Layer a = Layer
  { weights   :: SparseMatrix a
  , biases    :: SparseVector a
  } deriving Show

data Network a = Network
  { layers :: [Layer a]
  , sigma  :: a -> a
  , opT    :: a -> a
  }

vectorialise :: (Num a, Num b, Eq a, Eq b) =>
                (a -> b) -> SparseVector a -> SparseVector b
vectorialise f = sparseList . fmap f . fillVec

-- Hadamard product
hMult :: (Num a, Eq a) =>
                   SparseVector a -> SparseVector a -> SparseVector a
hMult = intersectVecsWith (*)

-- Input and output are collumn vectors of values

type Input a = SparseVector a
type Output a = SparseVector a

-- Assert that layers is a non empty list
forward :: (Num a, Eq a) => Input a -> Network a
        -> ([SparseVector a], [SparseVector a])
forward input net = foldl acc ([input], []) coefsList
  where
    coefsList = map getCoefs . layers $ net
    acc ((x:xs), ys) mat = (aVec : x : xs, zVec : ys)
      where
        (aVec, zVec) = step x mat
    step vec (mat, bs) = (aVec, zVec)
      where
        zVec = bs + (mat `mulMV` vec)
        aVec = vectorialise (sigma net) zVec
    getCoefs net = (weights net, biases net)

eval :: (Num a, Eq a) => Input a -> Network a -> Output a
eval input net = head . fst $ forward input net

learn :: (Num a, Eq a) =>
         a -> Input a -> Output a -> Network a -> Network a
learn h x y net = net {layers = map correctLayer (zip3 omegaList betaList (layers net))}
  where
    correctLayer (omega, beta, layer) = layer
      { weights = fmap (opT net) $ (weights layer) - fmap (*h) omega
      , biases  = fmap (opT net) $ (biases layer)  - fmap (*h) beta
      }
    -- -> order
    betaList  = tail . reverse $ deltaList
    -- -> order
    omegaList = reverse $ map buildOmega $ zip deltaList as
    buildOmega (aVec, deltaVec) = fromAssocListWithSize (dim aVec, dim deltaVec) matList
      where
        -- <- order
        matList = [((k, j), a * delta) |
                   (k, a) <- vecToAssocList aVec,
                   (j,delta) <- vecToAssocList deltaVec]
    (aL : as, zs) = forward x net
    deltaList = scanl (flip $ uncurry computeDelta) deltaL wAndA
    computeDelta matL al deltaL = (trans matL `mulMV` deltaL)
                                  `hMult` (vectorialise sigma' aL)
    -- <- order
    deltaL = nabla_aC `hMult` (vectorialise sigma' aL)
    nabla_aC = aL - y
    -- sigmoid derivative aplied to a_j^L
    sigma' a = a * (1 - a)
    wAndA = ws `zip` as
    ws = fmap weights . reverse . layers $ net

emptyLayer :: (Num a, Eq a) => Int -> Int -> Layer a
emptyLayer inputSize nbNeurons = Layer
  { weights = zeroMx (nbNeurons, inputSize)
  , biases = zeroVec nbNeurons
  }

randomLayer :: (Num a, Eq a) => [a] -> Int -> Int -> Layer a
randomLayer rList inputSize nbNeurons = Layer
  { weights = sparseMx mat
  , biases = sparseList lb
  }
  where
    (lb, rList') = splitAt nbNeurons rList
    splitMat l = row : (splitMat l')
      where
      (row, l') = splitAt inputSize rList
    mat = take nbNeurons $ splitMat rList

sigmoid :: Floating a => a -> a
sigmoid x = 1 / (1 + (exp $ -x))

-- DIGIT NeuralNetwork
layerHidden rList = randomLayer rList 784 15 
layerOutput rList = randomLayer rList 15 10
networkDigits rList = Network [layerHidden rList, layerOutput rList] sigmoid id

-- Teste


-- layer_a = emptyLayer 4 2
-- layer_b = emptyLayer 6 4
-- net = Network [layer_a, layer_b] sigmoid (\x -> if abs x > 0.2 then x - 0.02 * signum x else 0)
-- input = sparseList [32.0, 4.0]
-- output = sparseList [0.8, 0.7, 0.2, 0.8, 0.4, 0.2]


--     *
-- *   *   *
-- *   *   *
-- I   La  Lb

