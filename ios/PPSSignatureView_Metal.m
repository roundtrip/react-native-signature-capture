#import "PPSSignatureView_Metal.h"
#import "RSSignatureViewManager.h"
#import <simd/simd.h>

#define             STROKE_WIDTH_MIN 0.004 // Stroke width determined by touch velocity
#define             STROKE_WIDTH_MAX 0.030
#define       STROKE_WIDTH_SMOOTHING 0.5   // Low pass filter alpha

#define           VELOCITY_CLAMP_MIN 20
#define           VELOCITY_CLAMP_MAX 5000

#define QUADRATIC_DISTANCE_TOLERANCE 3.0   // Minimum distance to make a curve

#define             MAXIMUM_VERTECES 100000

static simd_float3 StrokeColor = { 0, 0, 0 };
static simd_float4 clearColor = { 1, 1, 1, 0 };

// Vertex structure containing 3D point and color
struct PPSSignaturePoint
{
    simd_float3     vertex;
    simd_float3     color;
};
typedef struct PPSSignaturePoint PPSSignaturePoint;

struct Uniforms {
    simd_float4x4 projectionMatrix;
    simd_float4x4 modelViewMatrix;
    simd_float4 constantColor;
};

// Maximum verteces in signature
static const int maxLength = MAXIMUM_VERTECES;

static inline CGPoint QuadraticPointInCurve(CGPoint start, CGPoint end, CGPoint controlPoint, float percent) {
    double a = pow((1.0 - percent), 2.0);
    double b = 2.0 * percent * (1.0 - percent);
    double c = pow(percent, 2.0);
    
    return (CGPoint) {
        a * start.x + b * controlPoint.x + c * end.x,
        a * start.y + b * controlPoint.y + c * end.y
    };
}

static float generateRandom(float from, float to) { return random() % 10000 / 10000.0 * (to - from) + from; }
static float clamp(float min, float max, float value) { return fmaxf(min, fminf(max, value)); }

// Find perpendicular vector from two other vectors to compute triangle strip around line
static simd_float3 perpendicular(PPSSignaturePoint p1, PPSSignaturePoint p2) {
    simd_float3 ret;
    ret.x = p2.vertex.y - p1.vertex.y;
    ret.y = -1 * (p2.vertex.x - p1.vertex.x);
    ret.z = 0;
    return ret;
}

static PPSSignaturePoint ViewPointToGL(CGPoint viewPoint, CGRect bounds, simd_float3 color) {
    
    return (PPSSignaturePoint) {
        {
            (viewPoint.x / bounds.size.width * 2.0 - 1),
            ((viewPoint.y / bounds.size.height) * 2.0 - 1) * -1,
            0
        },
        color
    };
}

static simd_float4x4 matrix_orthographic(float left, float right, float bottom, float top, float near, float far) {
    float rl = right - left;
    float tb = top - bottom;
    float fn = far - near;
    
    simd_float4x4 m = matrix_identity_float4x4;
    m.columns[0][0] = 2.0f / rl;
    m.columns[1][1] = 2.0f / tb;
    m.columns[2][2] = -2.0f / fn;
    m.columns[3][0] = -(right + left) / rl;
    m.columns[3][1] = -(top + bottom) / tb;
    m.columns[3][2] = -(far + near) / fn;
    
    return m;
}

static simd_float4x4 matrix_translation(float tx, float ty, float tz) {
    simd_float4x4 m = matrix_identity_float4x4;
    m.columns[3][0] = tx;
    m.columns[3][1] = ty;
    m.columns[3][2] = tz;
    return m;
}

@interface PPSSignatureView_Metal () <MTKViewDelegate> {
    // Metal state
    id<MTLDevice> device;
    id<MTLCommandQueue> commandQueue;
    id<MTLRenderPipelineState> pipelineState;
    id<MTLDepthStencilState> depthStencilState;
    
    id<MTLBuffer> vertexBuffer;
    id<MTLBuffer> dotsBuffer;
    id<MTLBuffer> uniformsBuffer;
    
    // Array of verteces, with current length
    PPSSignaturePoint *SignatureVertexData;
    uint length;
    uint vertexBufferCapacity;
    
    PPSSignaturePoint *SignatureDotsData;
    uint dotsLength;
    uint dotsBufferCapacity;
    
    struct Uniforms uniforms;
    
    // Width of line at current and previous vertex
    float penThickness;
    float previousThickness;
    
    // Previous points for quadratic bezier computations
    CGPoint previousPoint;
    CGPoint previousMidPoint;
    PPSSignaturePoint previousVertex;
    PPSSignaturePoint currentVelocity;
    UIColor* backgroundColor;
    UIColor* strokeColor;
}

@end

@implementation PPSSignatureView_Metal

- (void)commonInit {
    device = MTLCreateSystemDefaultDevice();
    
    if (device) {
        time(NULL);
        
        self.backgroundColor = [UIColor whiteColor];
        self.strokeColor = [UIColor blackColor];
        self.opaque = NO;
        
        self.device = device;
        self.delegate = self;
        self.enableSetNeedsDisplay = YES;
        self.sampleCount = 1; // No multisampling for simplicity
        self.clearColor = MTLClearColorMake(clearColor.x, clearColor.y, clearColor.z, clearColor.w);
        
        [self setupMetal];
        
        // Capture touches
        UIPanGestureRecognizer *pan = [[UIPanGestureRecognizer alloc] initWithTarget:self action:@selector(pan:)];
        pan.maximumNumberOfTouches = pan.minimumNumberOfTouches = 1;
        pan.cancelsTouchesInView = YES;
        [self addGestureRecognizer:pan];
        
        // For dotting your i's
        UITapGestureRecognizer *tap = [[UITapGestureRecognizer alloc] initWithTarget:self action:@selector(tap:)];
        tap.cancelsTouchesInView = YES;
        [self addGestureRecognizer:tap];
        
        // Erase with long press
        UILongPressGestureRecognizer *longer = [[UILongPressGestureRecognizer alloc] initWithTarget:self action:@selector(longPress:)];
        longer.cancelsTouchesInView = YES;
        [self addGestureRecognizer:longer];
    }
    else
        [NSException raise:@"MetalDeviceException" format:@"Failed to create Metal device"];
}

- (id)initWithCoder:(NSCoder *)aDecoder {
    if (self = [super initWithCoder:aDecoder]) [self commonInit];
    return self;
}

- (id)initWithFrame:(CGRect)frame {
    if (self = [super initWithFrame:frame]) [self commonInit];
    return self;
}

- (void)dealloc {
    free(SignatureVertexData);
    free(SignatureDotsData);
}

- (void)setupMetal {
    if (!device) {
        NSLog(@"Metal device not initialized");
        return;
    }
    
    commandQueue = [device newCommandQueue];
    if (!commandQueue) {
        NSLog(@"Failed to create command queue");
        return;
    }
    
    // Load shaders from embedded source
    NSError *error = nil;
    
    NSString *shaderSource = @"#include <metal_stdlib>\n"
        "using namespace metal;\n"
        "\n"
        "struct VertexIn {\n"
        "    float3 position [[attribute(0)]];\n"
        "    float3 color [[attribute(1)]];\n"
        "};\n"
        "\n"
        "struct VertexOut {\n"
        "    float4 position [[position]];\n"
        "    float4 color;\n"
        "};\n"
        "\n"
        "struct Uniforms {\n"
        "    float4x4 projectionMatrix;\n"
        "    float4x4 modelViewMatrix;\n"
        "    float4 constantColor;\n"
        "};\n"
        "\n"
        "vertex VertexOut signatureVertexShader(VertexIn in [[stage_in]],\n"
        "                                       constant Uniforms &uniforms [[buffer(1)]]) {\n"
        "    VertexOut out;\n"
        "    \n"
        "    out.position = float4(in.position.xy, 0.0, 1.0);\n"
        "    out.color = uniforms.constantColor;\n"
        "    \n"
        "    return out;\n"
        "}\n"
        "\n"
        "fragment float4 signatureFragmentShader(VertexOut in [[stage_in]]) {\n"
        "    return in.color;\n"
        "}";
    
    id<MTLLibrary> library = [device newLibraryWithSource:shaderSource options:nil error:&error];
    
    if (!library) {
        NSLog(@"Failed to create Metal library from source: %@", error);
        return;
    }
    
    id<MTLFunction> vertexFunction = [library newFunctionWithName:@"signatureVertexShader"];
    id<MTLFunction> fragmentFunction = [library newFunctionWithName:@"signatureFragmentShader"];
    
    if (!vertexFunction || !fragmentFunction) {
        NSLog(@"Failed to load Metal shaders");
        return;
    }
    
    // Create vertex descriptor
    MTLVertexDescriptor *vertexDescriptor = [[MTLVertexDescriptor alloc] init];
    vertexDescriptor.attributes[0].format = MTLVertexFormatFloat3;
    vertexDescriptor.attributes[0].offset = offsetof(PPSSignaturePoint, vertex);
    vertexDescriptor.attributes[0].bufferIndex = 0;
    
    vertexDescriptor.attributes[1].format = MTLVertexFormatFloat3;
    vertexDescriptor.attributes[1].offset = offsetof(PPSSignaturePoint, color);
    vertexDescriptor.attributes[1].bufferIndex = 0;
    
    vertexDescriptor.layouts[0].stride = sizeof(PPSSignaturePoint);
    vertexDescriptor.layouts[0].stepFunction = MTLVertexStepFunctionPerVertex;
    
    // Create pipeline state
    MTLRenderPipelineDescriptor *pipelineDescriptor = [[MTLRenderPipelineDescriptor alloc] init];
    pipelineDescriptor.vertexFunction = vertexFunction;
    pipelineDescriptor.fragmentFunction = fragmentFunction;
    pipelineDescriptor.vertexDescriptor = vertexDescriptor;
    pipelineDescriptor.colorAttachments[0].pixelFormat = self.colorPixelFormat;
    pipelineDescriptor.sampleCount = self.sampleCount;
    
    // Enable blending for transparency
    pipelineDescriptor.colorAttachments[0].blendingEnabled = YES;
    pipelineDescriptor.colorAttachments[0].sourceRGBBlendFactor = MTLBlendFactorSourceAlpha;
    pipelineDescriptor.colorAttachments[0].destinationRGBBlendFactor = MTLBlendFactorOneMinusSourceAlpha;
    
    pipelineState = [device newRenderPipelineStateWithDescriptor:pipelineDescriptor error:&error];
    if (!pipelineState) {
        NSLog(@"Failed to create pipeline state: %@", error);
        return;
    }
    
    // Create depth stencil state (disabled for 2D rendering)
    MTLDepthStencilDescriptor *depthDescriptor = [[MTLDepthStencilDescriptor alloc] init];
    depthDescriptor.depthCompareFunction = MTLCompareFunctionAlways;
    depthDescriptor.depthWriteEnabled = NO;
    depthStencilState = [device newDepthStencilStateWithDescriptor:depthDescriptor];
    
    // Initialize buffers
    vertexBufferCapacity = maxLength;
    SignatureVertexData = (PPSSignaturePoint *)calloc(vertexBufferCapacity, sizeof(PPSSignaturePoint));
    vertexBuffer = [device newBufferWithLength:sizeof(PPSSignaturePoint) * vertexBufferCapacity
                                       options:MTLResourceStorageModeShared];
    
    dotsBufferCapacity = maxLength;
    SignatureDotsData = (PPSSignaturePoint *)calloc(dotsBufferCapacity, sizeof(PPSSignaturePoint));
    dotsBuffer = [device newBufferWithLength:sizeof(PPSSignaturePoint) * dotsBufferCapacity
                                     options:MTLResourceStorageModeShared];
    
    // Setup uniforms
    uniforms.projectionMatrix = matrix_orthographic(-1, 1, -1, 1, 0.1f, 2.0f);
    uniforms.modelViewMatrix = matrix_translation(0.0f, 0.0f, -1.0f);
    [self updateStrokeColor];
    
    uniformsBuffer = [device newBufferWithBytes:&uniforms
                                          length:sizeof(struct Uniforms)
                                         options:MTLResourceStorageModeShared];
    
    length = 0;
    dotsLength = 0;
    penThickness = 0.003;
    previousPoint = CGPointMake(-100, -100);
}

- (void)updateBuffers {
    // Update vertex buffer
    if (length > 0 && SignatureVertexData && vertexBuffer) {
        memcpy([vertexBuffer contents], SignatureVertexData, sizeof(PPSSignaturePoint) * length);
    }
    
    // Update dots buffer
    if (dotsLength > 0 && SignatureDotsData && dotsBuffer) {
        memcpy([dotsBuffer contents], SignatureDotsData, sizeof(PPSSignaturePoint) * dotsLength);
    }
    
    // Update uniforms
    if (uniformsBuffer) {
        memcpy([uniformsBuffer contents], &uniforms, sizeof(struct Uniforms));
    }
}

- (void)addVertex:(PPSSignaturePoint)v toBuffer:(PPSSignaturePoint **)buffer length:(uint *)bufferLength capacity:(uint)capacity {
    if ((*bufferLength) >= capacity) {
        return;
    }
    
    (*buffer)[(*bufferLength)] = v;
    (*bufferLength)++;
}

- (void)mtkView:(MTKView *)view drawableSizeWillChange:(CGSize)size {
    // Handle view resize if needed
}

- (void)drawInMTKView:(MTKView *)view {
    if (!commandQueue || !pipelineState) {
        return;
    }
    
    id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
    if (!commandBuffer) {
        return;
    }
    
    MTLRenderPassDescriptor *renderPassDescriptor = view.currentRenderPassDescriptor;
    if (renderPassDescriptor == nil) {
        return;
    }
    
    [self updateBuffers];
    
    id<MTLRenderCommandEncoder> renderEncoder = [commandBuffer renderCommandEncoderWithDescriptor:renderPassDescriptor];
    if (!renderEncoder) {
        return;
    }
    
    [renderEncoder setRenderPipelineState:pipelineState];
    if (depthStencilState) {
        [renderEncoder setDepthStencilState:depthStencilState];
    }
    
    // Draw signature lines
    if (length > 2) {
        [renderEncoder setVertexBuffer:vertexBuffer offset:0 atIndex:0];
        [renderEncoder setVertexBuffer:uniformsBuffer offset:0 atIndex:1];
        [renderEncoder drawPrimitives:MTLPrimitiveTypeTriangleStrip vertexStart:0 vertexCount:length];
    }
    
    // Draw dots
    if (dotsLength > 0) {
        [renderEncoder setVertexBuffer:dotsBuffer offset:0 atIndex:0];
        [renderEncoder setVertexBuffer:uniformsBuffer offset:0 atIndex:1];
        [renderEncoder drawPrimitives:MTLPrimitiveTypeTriangleStrip vertexStart:0 vertexCount:dotsLength];
    }
    
    [renderEncoder endEncoding];
    
    [commandBuffer presentDrawable:view.currentDrawable];
    [commandBuffer commit];
}

- (void)erase {
    length = 0;
    dotsLength = 0;
    self.hasSignature = NO;
    
    [self setNeedsDisplay];
}

- (UIImage*)imageByCombiningImage:(UIImage*)firstImage withImage:(UIImage*)secondImage {
    UIImage *image = nil;
    
    CGSize newImageSize = CGSizeMake(MAX(firstImage.size.width, secondImage.size.width), MAX(firstImage.size.height, secondImage.size.height));
    if (UIGraphicsBeginImageContextWithOptions != NULL) {
        UIGraphicsBeginImageContextWithOptions(newImageSize, NO, [[UIScreen mainScreen] scale]);
    } else {
        UIGraphicsBeginImageContext(newImageSize);
    }
    [firstImage drawAtPoint:CGPointMake(roundf((newImageSize.width-firstImage.size.width)/2),
                                        roundf((newImageSize.height-firstImage.size.height)/2))];
    [secondImage drawAtPoint:CGPointMake(roundf((newImageSize.width-secondImage.size.width)/2),
                                         roundf((newImageSize.height-secondImage.size.height)/2))];
    image = UIGraphicsGetImageFromCurrentImageContext();
    UIGraphicsEndImageContext();
    
    return image;
}

- (UIImage *)snapshot {
    CGSize drawableSize = self.drawableSize;
    if (drawableSize.width == 0 || drawableSize.height == 0) {
        return nil;
    }
    
    // Create a temporary texture to render into
    MTLTextureDescriptor *textureDescriptor = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatBGRA8Unorm
                                                                                                 width:drawableSize.width
                                                                                                height:drawableSize.height
                                                                                             mipmapped:NO];
    textureDescriptor.usage = MTLTextureUsageRenderTarget;
    id<MTLTexture> offscreenTexture = [device newTextureWithDescriptor:textureDescriptor];
    
    // Create command buffer
    id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
    
    // Create render pass descriptor for offscreen rendering
    MTLRenderPassDescriptor *renderPass = [MTLRenderPassDescriptor renderPassDescriptor];
    renderPass.colorAttachments[0].texture = offscreenTexture;
    renderPass.colorAttachments[0].loadAction = MTLLoadActionClear;
    renderPass.colorAttachments[0].storeAction = MTLStoreActionStore;
    renderPass.colorAttachments[0].clearColor = MTLClearColorMake(clearColor.x, clearColor.y, clearColor.z, clearColor.w);
    
    // Render to offscreen texture
    id<MTLRenderCommandEncoder> renderEncoder = [commandBuffer renderCommandEncoderWithDescriptor:renderPass];
    
    [renderEncoder setRenderPipelineState:pipelineState];
    if (depthStencilState) {
        [renderEncoder setDepthStencilState:depthStencilState];
    }
    
    // Update buffers before rendering
    [self updateBuffers];
    
    // Draw signature lines
    if (length > 2) {
        [renderEncoder setVertexBuffer:vertexBuffer offset:0 atIndex:0];
        [renderEncoder setVertexBuffer:uniformsBuffer offset:0 atIndex:1];
        [renderEncoder drawPrimitives:MTLPrimitiveTypeTriangleStrip vertexStart:0 vertexCount:length];
    }
    
    // Draw dots
    if (dotsLength > 0) {
        [renderEncoder setVertexBuffer:dotsBuffer offset:0 atIndex:0];
        [renderEncoder setVertexBuffer:uniformsBuffer offset:0 atIndex:1];
        [renderEncoder drawPrimitives:MTLPrimitiveTypeTriangleStrip vertexStart:0 vertexCount:dotsLength];
    }
    
    [renderEncoder endEncoding];
    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];
    
    // Convert Metal texture to UIImage
    CGSize imageSize = CGSizeMake(offscreenTexture.width, offscreenTexture.height);
    NSUInteger bytesPerRow = 4 * offscreenTexture.width;
    NSUInteger totalBytes = bytesPerRow * offscreenTexture.height;
    void *imageBytes = malloc(totalBytes);
    
    [offscreenTexture getBytes:imageBytes
                   bytesPerRow:bytesPerRow
                    fromRegion:MTLRegionMake2D(0, 0, offscreenTexture.width, offscreenTexture.height)
                   mipmapLevel:0];
    
    CGColorSpaceRef colorSpace = CGColorSpaceCreateDeviceRGB();

    // Create NSData to properly manage the buffer lifecycle
    NSData *imageData = [NSData dataWithBytes:imageBytes length:totalBytes];
    free(imageBytes); // Free the malloc'd buffer since NSData has copied it

    CGDataProviderRef dataProvider = CGDataProviderCreateWithCFData((__bridge CFDataRef)imageData);

    CGImageRef cgImage = CGImageCreate(imageSize.width, imageSize.height,
                                      8, 32, bytesPerRow,
                                      colorSpace,
                                      kCGImageAlphaFirst | kCGBitmapByteOrder32Little,
                                      dataProvider, NULL, false,
                                      kCGRenderingIntentDefault);

    UIImage *image = [UIImage imageWithCGImage:cgImage];

    // Cleanup
    CGImageRelease(cgImage);
    CGDataProviderRelease(dataProvider);
    CGColorSpaceRelease(colorSpace);
    
    return image;
}

- (UIImage*)rotateImage:(UIImage*)sourceImage clockwise:(BOOL)clockwise {
    CGSize size = sourceImage.size;
    UIGraphicsBeginImageContext(CGSizeMake(size.height, size.width));
    [[UIImage imageWithCGImage:[sourceImage CGImage]
                         scale:1.0
                   orientation:clockwise ? UIImageOrientationRight : UIImageOrientationLeft]
     drawInRect:CGRectMake(0,0,size.height ,size.width)];
    
    UIImage* newImage = UIGraphicsGetImageFromCurrentImageContext();
    UIGraphicsEndImageContext();
    
    return newImage;
}

- (UIImage*) reduceImage:(UIImage*)image toSize:(CGSize)newSize {
    CGSize scaledSize = newSize;
    float scaleFactor = 1.0;
    
    if(image.size.width > image.size.height) {
        scaleFactor = image.size.width / image.size.height;
        scaledSize.width = newSize.width;
        scaledSize.height = newSize.height / scaleFactor;
    }
    else {
        scaleFactor = image.size.height / image.size.width;
        scaledSize.height = newSize.height;
        scaledSize.width = newSize.width / scaleFactor;
    }
    
    NSLog(@"%f x %f", scaledSize.width, scaledSize.height);
    
    UIGraphicsBeginImageContext(scaledSize);
    CGRect scaledImageRect = CGRectMake( 0.0, 0.0, scaledSize.width, scaledSize.height );
    [image drawInRect:scaledImageRect];
    
    UIImage* scaledImage = UIGraphicsGetImageFromCurrentImageContext();
    UIGraphicsEndImageContext();
    
    return scaledImage;
}

- (UIImage *)signatureImage {
    return [self signatureImage:false withSquare:false];
}

- (UIImage *)signatureImage: (BOOL) rotatedImage {
    return [self signatureImage:rotatedImage withSquare:false];
}

- (UIImage *)signatureImage: (BOOL) rotatedImage withSquare:(BOOL) square {
    if (!self.hasSignature)
        return nil;
    
    UIImage *signatureImg;
    UIImage *snapshot = [self snapshot];
    [self erase];
    
    if ( UI_USER_INTERFACE_IDIOM() == UIUserInterfaceIdiomPad ) {
        //signature
        if (square) {
            signatureImg = [self reduceImage:snapshot toSize: CGSizeMake(400.0f, 400.0f)];
        }
        else {
            signatureImg = snapshot;
        }
    }
    else {
        //rotate iphone signature - iphone's signature screen is always landscape
        
        if (rotatedImage) {
            if (square) {
                UIImage *rotatedImg = [self rotateImage:snapshot clockwise:false];
                signatureImg = [self reduceImage:rotatedImg toSize: CGSizeMake(400.0f, 400.0f)];
            }
            else {
                UIImage *rotatedImg = [self rotateImage:snapshot clockwise:false];
                signatureImg = rotatedImg;
            }
        }
        else {
            if (square) {
                signatureImg = [self reduceImage:snapshot toSize: CGSizeMake(400.0f, 400.0f)];
            }
            else {
                signatureImg = snapshot;
            }
        }
    }
    
    return signatureImg;
}

#pragma mark - Gesture Recognizers

- (void)tap:(UITapGestureRecognizer *)t {
    CGPoint l = [t locationInView:self];
    
    if (t.state == UIGestureRecognizerStateRecognized) {
        PPSSignaturePoint touchPoint = ViewPointToGL(l, self.bounds, (simd_float3){1, 1, 1});
        [self addVertex:touchPoint toBuffer:&SignatureDotsData length:&dotsLength capacity:dotsBufferCapacity];
        
        PPSSignaturePoint centerPoint = touchPoint;
        centerPoint.color = StrokeColor;
        [self addVertex:centerPoint toBuffer:&SignatureDotsData length:&dotsLength capacity:dotsBufferCapacity];
        
        static int segments = 20;
        simd_float2 radius = (simd_float2){
            clamp(0.00001, 0.02, penThickness * generateRandom(0.5, 1.5)),
            clamp(0.00001, 0.02, penThickness * generateRandom(0.5, 1.5))
        };
        simd_float2 velocityRadius = radius;
        float angle = 0;
        
        for (int i = 0; i <= segments; i++) {
            PPSSignaturePoint p = centerPoint;
            p.vertex.x += velocityRadius.x * cosf(angle);
            p.vertex.y += velocityRadius.y * sinf(angle);
            
            [self addVertex:p toBuffer:&SignatureDotsData length:&dotsLength capacity:dotsBufferCapacity];
            [self addVertex:centerPoint toBuffer:&SignatureDotsData length:&dotsLength capacity:dotsBufferCapacity];
            
            angle += M_PI * 2.0 / segments;
        }
        
        [self addVertex:touchPoint toBuffer:&SignatureDotsData length:&dotsLength capacity:dotsBufferCapacity];
    }
    
    [self setNeedsDisplay];
}

- (void)longPress:(UILongPressGestureRecognizer *)lp {
    [self erase];
}

- (void)pan:(UIPanGestureRecognizer *)p {
    CGPoint v = [p velocityInView:self];
    CGPoint l = [p locationInView:self];
    
    currentVelocity = ViewPointToGL(v, self.bounds, (simd_float3){0,0,0});
    float distance = 0.;
    if (previousPoint.x > 0) {
        distance = sqrtf((l.x - previousPoint.x) * (l.x - previousPoint.x) + (l.y - previousPoint.y) * (l.y - previousPoint.y));
    }
    
    float velocityMagnitude = sqrtf(v.x*v.x + v.y*v.y);
    float clampedVelocityMagnitude = clamp(VELOCITY_CLAMP_MIN, VELOCITY_CLAMP_MAX, velocityMagnitude);
    float normalizedVelocity = (clampedVelocityMagnitude - VELOCITY_CLAMP_MIN) / (VELOCITY_CLAMP_MAX - VELOCITY_CLAMP_MIN);
    
    float lowPassFilterAlpha = STROKE_WIDTH_SMOOTHING;
    float newThickness = (STROKE_WIDTH_MAX - STROKE_WIDTH_MIN) * (1 - normalizedVelocity) + STROKE_WIDTH_MIN;
    penThickness = penThickness * lowPassFilterAlpha + newThickness * (1 - lowPassFilterAlpha);
    
    if ([p state] == UIGestureRecognizerStateBegan) {
        previousPoint = l;
        previousMidPoint = l;
        
        PPSSignaturePoint startPoint = ViewPointToGL(l, self.bounds, (simd_float3){1, 1, 1});
        previousVertex = startPoint;
        previousThickness = penThickness;
        
        [self addVertex:startPoint toBuffer:&SignatureVertexData length:&length capacity:vertexBufferCapacity];
        [self addVertex:previousVertex toBuffer:&SignatureVertexData length:&length capacity:vertexBufferCapacity];
        
        self.hasSignature = YES;
        [self.manager publishDraggedEvent];
        
    } else if ([p state] == UIGestureRecognizerStateChanged) {
        CGPoint mid = CGPointMake((l.x + previousPoint.x) / 2.0, (l.y + previousPoint.y) / 2.0);
        
        if (distance > QUADRATIC_DISTANCE_TOLERANCE) {
            // Plot quadratic bezier instead of line
            unsigned int i;
            
            int segments = (int) distance / 1.5;
            
            float startPenThickness = previousThickness;
            float endPenThickness = penThickness;
            previousThickness = penThickness;
            
            for (i = 0; i < segments; i++) {
                penThickness = startPenThickness + ((endPenThickness - startPenThickness) / segments) * i;
                
                CGPoint quadPoint = QuadraticPointInCurve(previousMidPoint, mid, previousPoint, (float)i / (float)(segments));
                
                PPSSignaturePoint v = ViewPointToGL(quadPoint, self.bounds, StrokeColor);
                [self addTriangleStripPointsForPrevious:previousVertex next:v];
                
                previousVertex = v;
            }
        } else if (distance > 1.0) {
            PPSSignaturePoint v = ViewPointToGL(l, self.bounds, StrokeColor);
            [self addTriangleStripPointsForPrevious:previousVertex next:v];
            
            previousVertex = v;
            previousThickness = penThickness;
        }
        
        previousPoint = l;
        previousMidPoint = mid;
        
    } else if (p.state == UIGestureRecognizerStateEnded | p.state == UIGestureRecognizerStateCancelled) {
        PPSSignaturePoint v = ViewPointToGL(l, self.bounds, (simd_float3){1, 1, 1});
        [self addVertex:v toBuffer:&SignatureVertexData length:&length capacity:vertexBufferCapacity];
        
        previousVertex = v;
        [self addVertex:previousVertex toBuffer:&SignatureVertexData length:&length capacity:vertexBufferCapacity];
    }
    
    [self setNeedsDisplay];
}

- (void)setStrokeColor:(UIColor *)strokeColor {
    _strokeColor = strokeColor;
    [self updateStrokeColor];
}

#pragma mark - Private

- (void)updateStrokeColor {
    CGFloat red, green, blue, alpha, white;
    if (self.strokeColor && [self.strokeColor getRed:&red green:&green blue:&blue alpha:&alpha]) {
        uniforms.constantColor = simd_make_float4(red, green, blue, alpha);
        StrokeColor = simd_make_float3(red, green, blue);
    } else if (self.strokeColor && [self.strokeColor getWhite:&white alpha:&alpha]) {
        uniforms.constantColor = simd_make_float4(white, white, white, alpha);
        StrokeColor = simd_make_float3(white, white, white);
    } else {
        uniforms.constantColor = simd_make_float4(0, 0, 0, 1);
        StrokeColor = simd_make_float3(0, 0, 0);
    }
}

- (void)setBackgroundColor:(UIColor *)backgroundColor {
    [super setBackgroundColor:backgroundColor];
    
    CGFloat red, green, blue, alpha, white;
    if ([backgroundColor getRed:&red green:&green blue:&blue alpha:&alpha]) {
        clearColor = simd_make_float4(red, green, blue, alpha);
    } else if ([backgroundColor getWhite:&white alpha:&alpha]) {
        clearColor = simd_make_float4(white, white, white, alpha);
    }
    
    self.clearColor = MTLClearColorMake(clearColor.x, clearColor.y, clearColor.z, clearColor.w);
}

- (void)addTriangleStripPointsForPrevious:(PPSSignaturePoint)previous next:(PPSSignaturePoint)next {
    float toTravel = penThickness / 2.0;
    
    for (int i = 0; i < 2; i++) {
        simd_float3 p = perpendicular(previous, next);
        simd_float3 p1 = next.vertex;
        simd_float3 ref = p1 + p;
        
        float distance = simd_distance(p1, ref);
        float difX = p1.x - ref.x;
        float difY = p1.y - ref.y;
        float ratio = -1.0 * (toTravel / distance);
        
        difX = difX * ratio;
        difY = difY * ratio;
        
        PPSSignaturePoint stripPoint = {
            { p1.x + difX, p1.y + difY, 0.0 },
            StrokeColor
        };
        [self addVertex:stripPoint toBuffer:&SignatureVertexData length:&length capacity:vertexBufferCapacity];
        
        toTravel *= -1;
    }
}

@end