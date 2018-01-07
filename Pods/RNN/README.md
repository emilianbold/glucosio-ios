ios-RNN
=================

Simple Recurrent Neural Network that familiar with time series analysis, this RNN implemented 3 layers (Input, Hidden, Output) and Full-BPTT.

#### Podfile

```ruby
platform :ios, '9.0'
pod "RNN", "~> 1.0"
```

## How to use

#### Import
``` objective-c
#import "RNN.h"
```

#### Using GM1N model
``` objective-c
RNN *rnn = [[RNN alloc] init];

rnn.maxIteration     = 500;
rnn.convergenceError = 0.001f;
rnn.learningRate     = 0.5f;
rnn.timestepSize     = kRNNFullBPTT;

rnn.randomMax        = 0.25f;
rnn.randomMin        = -0.25f;

[rnn addPatternsFromArray:patterns];

[rnn createHiddenLayerNetsForCount:18];
[rnn createOutputLayerNetsForCount:10];

[rnn randomizeWeights];
[rnn uniformActiviation:RNNNetActivationSigmoid];

RNNOptimization *optimization = [[RNNOptimization alloc] init];
optimization.method           = RNNOptimizationStandardSGD;
[rnn uniformOptimization:optimization];

[rnn trainingWithIteration:^(NSInteger iteration, RNN *network) {
    NSLog(@"Iteration %li cost %lf", network.iteration, network.costFunction.mse);
} completion:^(NSInteger totalIteration, RNN *network) {
    [network predicateWithPatterns:patterns completion:^(NSArray<RNNSequenceOutput *> *sequenceOutputs) {
        [sequenceOutputs enumerateObjectsUsingBlock:^(RNNSequenceOutput * _Nonnull output, NSUInteger idx, BOOL * _Nonnull stop) {
            NSLog(@"(2) Predicated the %li outputs %@", idx, output.networkOutputs);
        }];
    }];
}];

```

#### How to Save / Fetch / Remove Trained Nework
``` objective-c
RNNFetcher *fetcher = [RNNFetcher sharedFetcher];

// Save RNN.
[fetcher save:rnn forKey:@"save1"];

// Fetch saved RNN.
RNN *nn = [fetcher objectForKey:@"save1"];

// Remove saved RNN.
[fetcher removeForKey:@"save1"];
```

## Todolist

1. RMSProp <br />
2. Adam <br />
3. Nadam <br />

## Version

V1.0

## License

MIT.
