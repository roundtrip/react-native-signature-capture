#import <UIKit/UIKit.h>
#import <MetalKit/MetalKit.h>

@class RSSignatureViewManager;

@interface PPSSignatureView_Metal : MTKView

@property (assign, nonatomic) UIColor *strokeColor;
@property (assign, nonatomic) BOOL hasSignature;
@property (strong, nonatomic) UIImage *signatureImage;
@property (nonatomic, strong) RSSignatureViewManager *manager;

- (void)erase;

- (UIImage *) signatureImage;
- (UIImage *) signatureImage: (BOOL) rotatedImage;
- (UIImage *) signatureImage: (BOOL) rotatedImage withSquare:(BOOL)square;

@end