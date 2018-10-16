# fibsem-optflow
Repository related to optical flow calculations for fibsem images


## Requirements

opencv

## Basic build
```
g++ -ggdb optflow.cpp features.cpp -o optflow `pkg-config --cflags --libs opencv` -lgsl -lblas -lpng -std=c++11
```
## Hints

Basic resin should use dual TV-L1. The sparse one isn't well configured yet either.
