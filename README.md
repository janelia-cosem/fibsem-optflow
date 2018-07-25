# fibsem-optflow
Repository related to optical flow calculations for fibsem images

## Basic build

g++ -ggdb optflow.cpp -o optflow `pkg-config --cflags --libs opencv` -lgsl -lblas

## Hints

Basic resin should use dual TV-L1. The sparse one isn't well configured yet either.
