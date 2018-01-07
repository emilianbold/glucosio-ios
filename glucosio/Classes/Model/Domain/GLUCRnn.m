#import <Foundation/Foundation.h>
#import "GLUCRnn.h"

@interface GLUCRnn ()
@end

@implementation GLUCRnn

+ (NSString *)title {
    return GLUCLoc(@"RNN");
}

+ (NSDictionary *)schema {
    return @{
             kGLUCModelSchemaPropertiesKey : @{
                     kGLUCUserRNNConvergenceErrorPropertyKey : @{
                             kGLUCModelAttributeKey : kGLUCUserRNNConvergenceErrorPropertyKey,
                             kGLUCModelAttributeTitleKey : @"kGLUCUserRNNConvergenceErrorPropertyKey",
                             kGLUCModelAttributeTypeKey : @"NSNumber",
                             },
                     kGLUCUserRNNLearningRatePropertyKey : @{
                             kGLUCModelAttributeKey : kGLUCUserRNNLearningRatePropertyKey,
                             kGLUCModelAttributeTitleKey : @"kGLUCUserRNNLearningRatePropertyKey",
                             kGLUCModelAttributeTypeKey : @"NSNumber",
                             },
             }
    };
}

- (NSArray *) settingsProperties {
    return @[kGLUCUserRNNConvergenceErrorPropertyKey, kGLUCUserRNNLearningRatePropertyKey];
}

- (NSDictionary *) settingsPropertiesTitle {
    return @{
             kGLUCUserRNNConvergenceErrorPropertyKey : @"Convergence Error",
             kGLUCUserRNNLearningRatePropertyKey: @"Learning Rate"
             };
}

- (instancetype) init {
    if ((self = [super init]) != nil) {
        self.convergenceError = @(0.0001f);
        self.learningRate = @(0.5f);
    }
    return self;
}


@end
