//
//  RNNNetOutputs.h
//  RNN
//
//  Created by kalvar_lin on 2017/7/4.
//  Copyright © 2017年 Kalvar Lin, the Knowledge Intelligence (K.I.). All rights reserved.
//

#import <Foundation/Foundation.h>

@interface RNNNetOutputs : NSObject <NSCopying, NSCoding>

@property (nonatomic) NSMutableArray <NSNumber *> *sumInputs;    // sumInputs through time.
@property (nonatomic) NSMutableArray <NSNumber *> *outputValues; // Output values through time. (outputValues 的上一刻 object 就是 b[t-1][h])
@property (nonatomic, readonly) double lastSumInput;             // 當前刻的輸出或為在 Forward 運算時的上一刻輸入值
@property (nonatomic, readonly) double lastOutputValue;          // 當前刻的輸出或為在 Forward 運算時的上一刻輸出值, e.g. b[t][h] or b[t-1][h]
@property (nonatomic, readonly) double previousOutput;           // 取出上一刻的輸出值, e.g. b[t-1][h]
@property (nonatomic, readonly) double previousSumInput;         // 取出上一刻的輸入值, e.g. a[t-1][h]
@property (nonatomic, readonly) NSInteger previousIndex;         // Last moment index for outputValues and sumInputs.
@property (nonatomic, readonly) NSInteger outputsCount;          // Same as time length.

- (double)lastSumInput;
- (double)lastOutputValue;
- (void)refresh;
- (void)removeLastOutput;

@end

@interface RNNNetOutputs (NSCoding)

- (void)encodeObject:(id)object forKey:(NSString *)key;
- (id)decodeForKey:(NSString *)key;

@end
