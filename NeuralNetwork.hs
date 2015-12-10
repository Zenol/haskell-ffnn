module NeuralNetwork where

import Math.LinearAlgebra.Sparse.Matrix
import Math.LinearAlgebra.Sparse.Vector


data Layer a = Layer
  { nb_neuron :: Int
  , weights   :: SparseMatrix a
  } deriving Show

data Network a = Network
  { layers :: [Layer a]
  , sigma  :: a -> a
  , sigma' :: a -> a
  }

vectorialise :: (Num a, Num b, Eq a, Eq b) =>
                (a -> b) -> SparseVector a -> SparseVector b
vectorialise f = sparseList . fmap f . fillVec


-- Input and output are collumn vectors of values

type Input a = SparseVector a
type Output a = SparseVector a

-- Assert that layers is a non empty list
eval :: (Num a, Eq a) => Input a -> Network a -> Output a
eval input net = foldl step input matrixList
  where
    matrixList = map (weights) . layers $ net
    step vec mat = (vectorialise $ sigma net) $ mulMV mat vec

learn :: [Input a] -> Network a -> Network a
learn = undefined


-- Teste

sigmoid :: Floating a => a -> a
sigmoid x = 1 / (1 + (exp $ -x))

layer_a = Layer 3 (fromAssocList [((1, 1), 1), ((2, 2), 1)])
layer_b = Layer 1 (fromAssocList [((1, 1), 1), ((1, 2), 1), ((1, 3), 1)])
net = Network [layer_a, layer_b] sigmoid undefined
input = sparseList [1, 2.0]
