# fibsem-optflow
Repository related to optical flow calculations for fibsem images


## Requirements

opencv
opencv - non-free
cuda

## Basic build
```
g++ -ggdb optflow.cpp features.cpp -o optflow `pkg-config --cflags --libs opencv` -lgsl -lblas -lpng -ljsoncpp -lboost_iostreams -std=c++11
```
## Hints

Only TV L-1 is done. Sparse is too iffy on resin.
