#ifdef __OBJC__
#import <UIKit/UIKit.h>
#else
#ifndef FOUNDATION_EXPORT
#if defined(__cplusplus)
#define FOUNDATION_EXPORT extern "C"
#else
#define FOUNDATION_EXPORT extern
#endif
#endif
#endif

#import "RNNActivation.h"
#import "RNNCost.h"
#import "RNNFetcher.h"
#import "RNNMath.h"
#import "RNNPartial.h"
#import "RNNNet.h"
#import "RNNNetOutputs.h"
#import "RNNOptimization.h"
#import "RNNTimestep.h"
#import "RNNPattern.h"
#import "RNNSequenceOutput.h"
#import "RNN.h"

FOUNDATION_EXPORT double RNNVersionNumber;
FOUNDATION_EXPORT const unsigned char RNNVersionString[];

