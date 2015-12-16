module MNIST where

import           Data.Word
import qualified Data.Vector as V
import           Data.Binary.Get
import qualified Data.ByteString as B
import qualified Data.ByteString.Lazy as LB

data ImageSet = ImageSet {
  magicIS          :: Word32,
  numberEntriesIS  :: Word32,
  width            :: Word32,
  height           :: Word32,
  images           :: [V.Vector Word8]
  } deriving Show

data LabelSet = LabelSet {
  magicLS          :: Word32,
  numberEntriesLS  :: Word32,
  labels           :: [Word8]
  } deriving Show

loadImagesGet :: Get ImageSet
loadImagesGet = do
  magic <- getWord32be
  n     <- getWord32be
  w     <- getWord32be
  h     <- getWord32be
  imgs  <- readList n $ getByteString (fromIntegral $ w*h)
  return $ ImageSet magic n w h (fmap bs2vec imgs)
    where
      bs2vec = V.fromList . B.unpack
      readList n = sequence . take (fromIntegral n) . repeat


loadLabelsGet :: Get LabelSet
loadLabelsGet = do
  magic <- getWord32be
  n     <- getWord32be
  ls    <- getLazyByteString (fromIntegral $ n)
  return $ LabelSet magic n (LB.unpack ls)

loadImageFile :: String -> IO ImageSet
loadImageFile fileName = do
  fmap (runGet loadImagesGet) (LB.readFile fileName)


loadLabelFile :: String -> IO LabelSet
loadLabelFile fileName = do
  fmap (runGet loadLabelsGet) (LB.readFile fileName)

loadMNIST :: String -> String -> IO [(V.Vector Word8, Word8)]
loadMNIST imageFile labelFile = do
  img <- loadImageFile imageFile
  lab <- loadLabelFile labelFile
  return $ zip (images img) (labels lab)
