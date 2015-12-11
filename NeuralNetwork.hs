module NeuralNetwork where

import Math.LinearAlgebra.Sparse.Matrix
import Math.LinearAlgebra.Sparse.Vector


data Layer a = Layer
  { nb_neuron :: Int
  , weights   :: SparseMatrix a
  , biases    :: SparseVector a
  } deriving Show

data Network a = Network
  { layers :: [Layer a]
  , sigma  :: a -> a
  , sigma' :: a -> a
  }

vectorialise :: (Num a, Num b, Eq a, Eq b) =>
                (a -> b) -> SparseVector a -> SparseVector b
vectorialise f = sparseList . fmap f . fillVec

-- Hadamard product
hMult :: (Num a, Num b, Eq a, Eq b) =>
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


-- learn :: (Input a, Output) -> Network a -> Network a
learn x y net = (deltaList, aList)
  where
    (aList@(aL : as), zs) = forward x net
    deltaList = scanl (flip $ uncurry computeDelta) deltaL wAndA
    computeDelta matL al deltaL = (trans matL `mulMV` deltaL)
                                  `hMult` (vectorialise sigma' aL)
    deltaL = nabla_aC `hMult` (vectorialise sigma' aL)
    nabla_aC = aL - y
    -- sigmoid derivative aplied to a_j^L
    sigma' a = a * (1 - a)
    wAndA = zip transMatrices as
    transMatrices = fmap (trans . weights) . reverse . layers $ net


-- Teste

sigmoid :: Floating a => a -> a
sigmoid x = 1 / (1 + (exp $ -x))

layer_a = Layer 3 (fromAssocList [((1, 1), 1), ((1, 2), 1), ((2, 2), 1)]) (sparseList [])
layer_b = Layer 1 (fromAssocList [((1, 1), 1), ((1, 2), 1), ((1, 3), 1)]) (sparseList [])
net = Network [layer_a, layer_b] sigmoid undefined
input = sparseList [1, 2.0]
