#import "GLUCModel.h"

@protocol RLMDouble;

static NSString *const kGLUCUserRNNConvergenceErrorPropertyKey = @"convergenceError";
static NSString *const kGLUCUserRNNLearningRatePropertyKey = @"learningRate";

@interface GLUCRnn : GLUCModel

@property (nonatomic, readwrite, strong) NSNumber<RLMDouble> *convergenceError;
@property (nonatomic, readwrite, strong) NSNumber<RLMDouble> *learningRate;

- (NSArray *) settingsProperties;
- (NSDictionary *) settingsPropertiesTitle;

@end
