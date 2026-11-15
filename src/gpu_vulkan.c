/* gpu_vulkan.c - Vulkan compute backend for matmuls (see gpu_vk.h).
 *
 * Mirrors gpu_d3d12.c with the same kernels: kernels_dx*.hlsl compiled to SPIR-V by
 * tools\vk_shaders.cmd (kernels_vk.h). Every weight matrix lives in its own device-local buffer
 * (Q6_K repacked to 212-byte blocks); vk_matmul copies the Q8_K activations into a GPU buffer,
 * dispatches one kernel per matrix and reads the results back, all in one command buffer.
 *
 * The D3D12 root signature (b0 constants, t0 t1, u0 u1 u2) becomes one descriptor set layout:
 * u0..u2 -> bindings 0..2, t0 t1 -> 8 9, b0 -> 16 (a uniform buffer: the constants of each
 * dispatch go to a slot of a host-visible ring). Every dispatch gets its own descriptor set;
 * the pool and the ring are reset after each submit. */
#include "gpu_vk.h"
#include "platform.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(PITHOS_UWP)

void vk_list_devices(void) { printf("  Vulkan not available on this platform\n"); }
Vulkan *vk_create(const Model *m, int device, char *info, int infolen) {
    (void)m; (void)device;
    snprintf(info, (size_t)infolen, "Vulkan not available on this platform");
    return NULL;
}
void vk_destroy(Vulkan *v) { (void)v; }
int vk_has(const Vulkan *v, const uint8_t *w) { (void)v; (void)w; return 0; }
int vk_matmul(Vulkan *v, const VulkanJob *jobs, int n) { (void)v; (void)jobs; (void)n; return 0; }
int vk_dense_init(Vulkan *v, const Model *m, int n_ctx, int n_batch, char *info, int infolen) {
    (void)v; (void)m; (void)n_ctx; (void)n_batch;
    snprintf(info, (size_t)infolen, "Vulkan not available on this platform");
    return 0;
}
int vk_dense_forward(Vulkan *v, const float *x, int n, const float *rope_cs, const float *rope_swa, int pos0,
                     float *logits, float *x_last) {
    (void)v; (void)x; (void)n; (void)rope_cs; (void)rope_swa; (void)pos0; (void)logits;
    return 0;
}
void vk_dense_kv(Vulkan *v, uint16_t *kc, uint16_t *vc, int r0, int r1, int to_gpu) {
    (void)v; (void)kc; (void)vc; (void)r0; (void)r1; (void)to_gpu;
}
void vk_dense_conv(Vulkan *v, float *conv, int to_gpu) { (void)v; (void)conv; (void)to_gpu; }
void vk_dense_set_moe(Vulkan *v, VkMoeFn fn, void *arg) { (void)v; (void)fn; (void)arg; }
int vk_experts_on(const Vulkan *v) { (void)v; return 0; }
int vk_expert_resident(const Vulkan *v, int layer, int expert) { (void)v; (void)layer; (void)expert; return 0; }
int vk_moe_ready(const Vulkan *v, int n) { (void)v; (void)n; return 0; }
int vk_moe_begin(Vulkan *v, int layer, const int *a_expert, const int *a_tok, const float *a_w, int n) {
    (void)v; (void)layer; (void)a_expert; (void)a_tok; (void)a_w; (void)n;
    return 0;
}
void vk_note_usage(Vulkan *v, int layer, int expert) { (void)v; (void)layer; (void)expert; }
void vk_rebalance(Vulkan *v, int max_uploads) { (void)v; (void)max_uploads; }

#else

#include <math.h>

#include "kernels_vk.h"

/* ------------------------------------------------------------------ minimal Vulkan ABI */

#if defined(_WIN32)
#define VKAPI __stdcall
#else
#define VKAPI
#endif

typedef int32_t VkResult;
typedef uint32_t VkFlags;
typedef uint32_t VkBool32;
typedef uint64_t VkDeviceSize;
typedef struct VkInstance_T *VkInstance;
typedef struct VkPhysicalDevice_T *VkPhysicalDevice;
typedef struct VkDevice_T *VkDevice;
typedef struct VkQueue_T *VkQueue;
typedef struct VkCommandBuffer_T *VkCommandBuffer;
/* non-dispatchable handles are 64-bit on every platform */
typedef uint64_t VkBuffer, VkDeviceMemory, VkShaderModule, VkPipeline, VkPipelineLayout, VkPipelineCache,
    VkDescriptorSetLayout, VkDescriptorPool, VkDescriptorSet, VkCommandPool, VkFence, VkSampler;

#define VK_SUCCESS 0
#define VK_NOT_READY 1
#define VK_TIMEOUT 2
#define VK_API_1_1 ((1u << 22) | (1u << 12))
#define VK_WHOLE_SIZE (~(VkDeviceSize)0)

enum {
    ST_APPLICATION_INFO = 0, ST_INSTANCE_CREATE_INFO = 1, ST_DEVICE_QUEUE_CREATE_INFO = 2, ST_DEVICE_CREATE_INFO = 3,
    ST_SUBMIT_INFO = 4, ST_MEMORY_ALLOCATE_INFO = 5, ST_FENCE_CREATE_INFO = 8, ST_BUFFER_CREATE_INFO = 12,
    ST_SHADER_MODULE_CREATE_INFO = 16, ST_PIPELINE_SHADER_STAGE_CREATE_INFO = 18, ST_COMPUTE_PIPELINE_CREATE_INFO = 29,
    ST_PIPELINE_LAYOUT_CREATE_INFO = 30, ST_DESCRIPTOR_SET_LAYOUT_CREATE_INFO = 32, ST_DESCRIPTOR_POOL_CREATE_INFO = 33,
    ST_DESCRIPTOR_SET_ALLOCATE_INFO = 34, ST_WRITE_DESCRIPTOR_SET = 35, ST_COMMAND_POOL_CREATE_INFO = 39,
    ST_COMMAND_BUFFER_ALLOCATE_INFO = 40, ST_COMMAND_BUFFER_BEGIN_INFO = 42, ST_MEMORY_BARRIER = 46
};

#define QUEUE_COMPUTE 0x2u
#define MEM_DEVICE_LOCAL 0x1u
#define MEM_HOST_VISIBLE 0x2u
#define MEM_HOST_COHERENT 0x4u
#define MEM_HOST_CACHED 0x8u
#define HEAP_DEVICE_LOCAL 0x1u
#define USAGE_TRANSFER_SRC 0x1u
#define USAGE_TRANSFER_DST 0x2u
#define USAGE_UNIFORM 0x10u
#define USAGE_STORAGE 0x20u
#define STAGE_COMPUTE_SHADER_BIT 0x20u
#define DESC_STORAGE_BUFFER 7
#define DESC_UNIFORM_DYNAMIC 8
#define BIND_COMPUTE 1
#define CB_ONE_TIME 0x1u
#define PS_TRANSFER 0x1000u
#define PS_COMPUTE 0x800u
#define PS_HOST 0x4000u
#define AC_UNIFORM_READ 0x8u
#define AC_SHADER_READ 0x20u
#define AC_SHADER_WRITE 0x40u
#define AC_TRANSFER_READ 0x800u
#define AC_TRANSFER_WRITE 0x1000u
#define AC_HOST_READ 0x2000u
#define AC_HOST_WRITE 0x4000u
#define PS_ALL 0x10000u
#define AC_MEMORY_READ 0x8000u
#define AC_MEMORY_WRITE 0x10000u

typedef struct {
    int sType; const void *pNext; const char *pApplicationName; uint32_t applicationVersion;
    const char *pEngineName; uint32_t engineVersion; uint32_t apiVersion;
} VkApplicationInfo;
typedef struct {
    int sType; const void *pNext; VkFlags flags; const VkApplicationInfo *pApplicationInfo;
    uint32_t enabledLayerCount; const char *const *ppEnabledLayerNames;
    uint32_t enabledExtensionCount; const char *const *ppEnabledExtensionNames;
} VkInstanceCreateInfo;

typedef struct {
    uint32_t maxImageDimension1D, maxImageDimension2D, maxImageDimension3D, maxImageDimensionCube, maxImageArrayLayers;
    uint32_t maxTexelBufferElements, maxUniformBufferRange, maxStorageBufferRange, maxPushConstantsSize;
    uint32_t maxMemoryAllocationCount, maxSamplerAllocationCount;
    VkDeviceSize bufferImageGranularity, sparseAddressSpaceSize;
    uint32_t maxBoundDescriptorSets, maxPerStageDescriptorSamplers, maxPerStageDescriptorUniformBuffers;
    uint32_t maxPerStageDescriptorStorageBuffers, maxPerStageDescriptorSampledImages, maxPerStageDescriptorStorageImages;
    uint32_t maxPerStageDescriptorInputAttachments, maxPerStageResources, maxDescriptorSetSamplers;
    uint32_t maxDescriptorSetUniformBuffers, maxDescriptorSetUniformBuffersDynamic, maxDescriptorSetStorageBuffers;
    uint32_t maxDescriptorSetStorageBuffersDynamic, maxDescriptorSetSampledImages, maxDescriptorSetStorageImages;
    uint32_t maxDescriptorSetInputAttachments, maxVertexInputAttributes, maxVertexInputBindings;
    uint32_t maxVertexInputAttributeOffset, maxVertexInputBindingStride, maxVertexOutputComponents;
    uint32_t maxTessellationGenerationLevel, maxTessellationPatchSize, maxTessellationControlPerVertexInputComponents;
    uint32_t maxTessellationControlPerVertexOutputComponents, maxTessellationControlPerPatchOutputComponents;
    uint32_t maxTessellationControlTotalOutputComponents, maxTessellationEvaluationInputComponents;
    uint32_t maxTessellationEvaluationOutputComponents, maxGeometryShaderInvocations, maxGeometryInputComponents;
    uint32_t maxGeometryOutputComponents, maxGeometryOutputVertices, maxGeometryTotalOutputComponents;
    uint32_t maxFragmentInputComponents, maxFragmentOutputAttachments, maxFragmentDualSrcAttachments;
    uint32_t maxFragmentCombinedOutputResources, maxComputeSharedMemorySize, maxComputeWorkGroupCount[3];
    uint32_t maxComputeWorkGroupInvocations, maxComputeWorkGroupSize[3], subPixelPrecisionBits, subTexelPrecisionBits;
    uint32_t mipmapPrecisionBits, maxDrawIndexedIndexValue, maxDrawIndirectCount;
    float maxSamplerLodBias, maxSamplerAnisotropy;
    uint32_t maxViewports, maxViewportDimensions[2];
    float viewportBoundsRange[2];
    uint32_t viewportSubPixelBits;
    size_t minMemoryMapAlignment;
    VkDeviceSize minTexelBufferOffsetAlignment, minUniformBufferOffsetAlignment, minStorageBufferOffsetAlignment;
    int32_t minTexelOffset; uint32_t maxTexelOffset; int32_t minTexelGatherOffset; uint32_t maxTexelGatherOffset;
    float minInterpolationOffset, maxInterpolationOffset;
    uint32_t subPixelInterpolationOffsetBits, maxFramebufferWidth, maxFramebufferHeight, maxFramebufferLayers;
    VkFlags framebufferColorSampleCounts, framebufferDepthSampleCounts, framebufferStencilSampleCounts;
    VkFlags framebufferNoAttachmentsSampleCounts;
    uint32_t maxColorAttachments;
    VkFlags sampledImageColorSampleCounts, sampledImageIntegerSampleCounts, sampledImageDepthSampleCounts;
    VkFlags sampledImageStencilSampleCounts, storageImageSampleCounts;
    uint32_t maxSampleMaskWords;
    VkBool32 timestampComputeAndGraphics;
    float timestampPeriod;
    uint32_t maxClipDistances, maxCullDistances, maxCombinedClipAndCullDistances, discreteQueuePriorities;
    float pointSizeRange[2], lineWidthRange[2], pointSizeGranularity, lineWidthGranularity;
    VkBool32 strictLines, standardSampleLocations;
    VkDeviceSize optimalBufferCopyOffsetAlignment, optimalBufferCopyRowPitchAlignment, nonCoherentAtomSize;
} VkPhysicalDeviceLimits;

typedef struct {
    uint32_t apiVersion, driverVersion, vendorID, deviceID;
    int deviceType;     /* 1 integrated, 2 discrete, 3 virtual, 4 CPU */
    char deviceName[256];
    uint8_t pipelineCacheUUID[16];
    VkPhysicalDeviceLimits limits;
    VkBool32 sparseProperties[5];
} VkPhysicalDeviceProperties;

/* the hand-written layouts must match the official headers (64-bit sizes) */
typedef char vk_limits_size_check[sizeof(void *) != 8 || sizeof(VkPhysicalDeviceLimits) == 504 ? 1 : -1];
typedef char vk_props_size_check[sizeof(void *) != 8 || sizeof(VkPhysicalDeviceProperties) == 824 ? 1 : -1];

typedef struct { VkFlags propertyFlags; uint32_t heapIndex; } VkMemoryType;
typedef struct { VkDeviceSize size; VkFlags flags; } VkMemoryHeap;
typedef struct {
    uint32_t memoryTypeCount; VkMemoryType memoryTypes[32];
    uint32_t memoryHeapCount; VkMemoryHeap memoryHeaps[16];
} VkPhysicalDeviceMemoryProperties;
typedef struct { VkFlags queueFlags; uint32_t queueCount, timestampValidBits, granularity[3]; } VkQueueFamilyProperties;

typedef struct {
    int sType; const void *pNext; VkFlags flags; uint32_t queueFamilyIndex, queueCount; const float *pQueuePriorities;
} VkDeviceQueueCreateInfo;
typedef struct {
    int sType; const void *pNext; VkFlags flags;
    uint32_t queueCreateInfoCount; const VkDeviceQueueCreateInfo *pQueueCreateInfos;
    uint32_t enabledLayerCount; const char *const *ppEnabledLayerNames;
    uint32_t enabledExtensionCount; const char *const *ppEnabledExtensionNames;
    const void *pEnabledFeatures;
} VkDeviceCreateInfo;
typedef struct { VkDeviceSize size, alignment; uint32_t memoryTypeBits; } VkMemoryRequirements;
typedef struct { int sType; const void *pNext; VkDeviceSize allocationSize; uint32_t memoryTypeIndex; } VkMemoryAllocateInfo;
typedef struct {
    int sType; const void *pNext; VkFlags flags; VkDeviceSize size; VkFlags usage; int sharingMode;
    uint32_t queueFamilyIndexCount; const uint32_t *pQueueFamilyIndices;
} VkBufferCreateInfo;
typedef struct { int sType; const void *pNext; VkFlags flags; size_t codeSize; const uint32_t *pCode; } VkShaderModuleCreateInfo;
typedef struct {
    int sType; const void *pNext; VkFlags flags; VkFlags stage; VkShaderModule module; const char *pName;
    const void *pSpecializationInfo;
} VkPipelineShaderStageCreateInfo;
typedef struct {
    int sType; const void *pNext; VkFlags flags; VkPipelineShaderStageCreateInfo stage; VkPipelineLayout layout;
    VkPipeline basePipelineHandle; int32_t basePipelineIndex;
} VkComputePipelineCreateInfo;
typedef struct {
    uint32_t binding; int descriptorType; uint32_t descriptorCount; VkFlags stageFlags; const VkSampler *pImmutableSamplers;
} VkDescriptorSetLayoutBinding;
typedef struct {
    int sType; const void *pNext; VkFlags flags; uint32_t bindingCount; const VkDescriptorSetLayoutBinding *pBindings;
} VkDescriptorSetLayoutCreateInfo;
typedef struct {
    int sType; const void *pNext; VkFlags flags; uint32_t setLayoutCount; const VkDescriptorSetLayout *pSetLayouts;
    uint32_t pushConstantRangeCount; const void *pPushConstantRanges;
} VkPipelineLayoutCreateInfo;
typedef struct { int type; uint32_t descriptorCount; } VkDescriptorPoolSize;
typedef struct {
    int sType; const void *pNext; VkFlags flags; uint32_t maxSets, poolSizeCount; const VkDescriptorPoolSize *pPoolSizes;
} VkDescriptorPoolCreateInfo;
typedef struct {
    int sType; const void *pNext; VkDescriptorPool descriptorPool; uint32_t descriptorSetCount;
    const VkDescriptorSetLayout *pSetLayouts;
} VkDescriptorSetAllocateInfo;
typedef struct { VkBuffer buffer; VkDeviceSize offset, range; } VkDescriptorBufferInfo;
typedef struct {
    int sType; const void *pNext; VkDescriptorSet dstSet; uint32_t dstBinding, dstArrayElement, descriptorCount;
    int descriptorType; const void *pImageInfo; const VkDescriptorBufferInfo *pBufferInfo; const void *pTexelBufferView;
} VkWriteDescriptorSet;
typedef struct { int sType; const void *pNext; VkFlags flags; uint32_t queueFamilyIndex; } VkCommandPoolCreateInfo;
typedef struct {
    int sType; const void *pNext; VkCommandPool commandPool; int level; uint32_t commandBufferCount;
} VkCommandBufferAllocateInfo;
typedef struct { int sType; const void *pNext; VkFlags flags; const void *pInheritanceInfo; } VkCommandBufferBeginInfo;
typedef struct { VkDeviceSize srcOffset, dstOffset, size; } VkBufferCopy;
typedef struct { int sType; const void *pNext; VkFlags srcAccessMask, dstAccessMask; } VkMemoryBarrier;
typedef struct {
    int sType; const void *pNext; uint32_t waitSemaphoreCount; const void *pWaitSemaphores; const VkFlags *pWaitDstStageMask;
    uint32_t commandBufferCount; const VkCommandBuffer *pCommandBuffers; uint32_t signalSemaphoreCount;
    const void *pSignalSemaphores;
} VkSubmitInfo;
typedef struct { int sType; const void *pNext; VkFlags flags; } VkFenceCreateInfo;

typedef void(VKAPI *PFN_void)(void);
typedef struct {
    PFN_void(VKAPI *GetInstanceProcAddr)(VkInstance, const char *);
    VkResult(VKAPI *CreateInstance)(const VkInstanceCreateInfo *, const void *, VkInstance *);
    void(VKAPI *DestroyInstance)(VkInstance, const void *);
    VkResult(VKAPI *EnumeratePhysicalDevices)(VkInstance, uint32_t *, VkPhysicalDevice *);
    void(VKAPI *GetPhysicalDeviceProperties)(VkPhysicalDevice, VkPhysicalDeviceProperties *);
    void(VKAPI *GetPhysicalDeviceMemoryProperties)(VkPhysicalDevice, VkPhysicalDeviceMemoryProperties *);
    void(VKAPI *GetPhysicalDeviceQueueFamilyProperties)(VkPhysicalDevice, uint32_t *, VkQueueFamilyProperties *);
    VkResult(VKAPI *CreateDevice)(VkPhysicalDevice, const VkDeviceCreateInfo *, const void *, VkDevice *);
    PFN_void(VKAPI *GetDeviceProcAddr)(VkDevice, const char *);
    /* device level */
    void(VKAPI *DestroyDevice)(VkDevice, const void *);
    void(VKAPI *GetDeviceQueue)(VkDevice, uint32_t, uint32_t, VkQueue *);
    VkResult(VKAPI *CreateBuffer)(VkDevice, const VkBufferCreateInfo *, const void *, VkBuffer *);
    void(VKAPI *DestroyBuffer)(VkDevice, VkBuffer, const void *);
    void(VKAPI *GetBufferMemoryRequirements)(VkDevice, VkBuffer, VkMemoryRequirements *);
    VkResult(VKAPI *AllocateMemory)(VkDevice, const VkMemoryAllocateInfo *, const void *, VkDeviceMemory *);
    void(VKAPI *FreeMemory)(VkDevice, VkDeviceMemory, const void *);
    VkResult(VKAPI *BindBufferMemory)(VkDevice, VkBuffer, VkDeviceMemory, VkDeviceSize);
    VkResult(VKAPI *MapMemory)(VkDevice, VkDeviceMemory, VkDeviceSize, VkDeviceSize, VkFlags, void **);
    VkResult(VKAPI *CreateShaderModule)(VkDevice, const VkShaderModuleCreateInfo *, const void *, VkShaderModule *);
    void(VKAPI *DestroyShaderModule)(VkDevice, VkShaderModule, const void *);
    VkResult(VKAPI *CreateDescriptorSetLayout)(VkDevice, const VkDescriptorSetLayoutCreateInfo *, const void *,
                                               VkDescriptorSetLayout *);
    void(VKAPI *DestroyDescriptorSetLayout)(VkDevice, VkDescriptorSetLayout, const void *);
    VkResult(VKAPI *CreatePipelineLayout)(VkDevice, const VkPipelineLayoutCreateInfo *, const void *, VkPipelineLayout *);
    void(VKAPI *DestroyPipelineLayout)(VkDevice, VkPipelineLayout, const void *);
    VkResult(VKAPI *CreateComputePipelines)(VkDevice, VkPipelineCache, uint32_t, const VkComputePipelineCreateInfo *,
                                            const void *, VkPipeline *);
    void(VKAPI *DestroyPipeline)(VkDevice, VkPipeline, const void *);
    VkResult(VKAPI *CreateDescriptorPool)(VkDevice, const VkDescriptorPoolCreateInfo *, const void *, VkDescriptorPool *);
    void(VKAPI *DestroyDescriptorPool)(VkDevice, VkDescriptorPool, const void *);
    VkResult(VKAPI *ResetDescriptorPool)(VkDevice, VkDescriptorPool, VkFlags);
    VkResult(VKAPI *AllocateDescriptorSets)(VkDevice, const VkDescriptorSetAllocateInfo *, VkDescriptorSet *);
    void(VKAPI *UpdateDescriptorSets)(VkDevice, uint32_t, const VkWriteDescriptorSet *, uint32_t, const void *);
    VkResult(VKAPI *CreateCommandPool)(VkDevice, const VkCommandPoolCreateInfo *, const void *, VkCommandPool *);
    void(VKAPI *DestroyCommandPool)(VkDevice, VkCommandPool, const void *);
    VkResult(VKAPI *ResetCommandPool)(VkDevice, VkCommandPool, VkFlags);
    VkResult(VKAPI *AllocateCommandBuffers)(VkDevice, const VkCommandBufferAllocateInfo *, VkCommandBuffer *);
    VkResult(VKAPI *BeginCommandBuffer)(VkCommandBuffer, const VkCommandBufferBeginInfo *);
    VkResult(VKAPI *EndCommandBuffer)(VkCommandBuffer);
    void(VKAPI *CmdBindPipeline)(VkCommandBuffer, int, VkPipeline);
    void(VKAPI *CmdBindDescriptorSets)(VkCommandBuffer, int, VkPipelineLayout, uint32_t, uint32_t, const VkDescriptorSet *,
                                       uint32_t, const uint32_t *);
    void(VKAPI *CmdDispatch)(VkCommandBuffer, uint32_t, uint32_t, uint32_t);
    void(VKAPI *CmdCopyBuffer)(VkCommandBuffer, VkBuffer, VkBuffer, uint32_t, const VkBufferCopy *);
    void(VKAPI *CmdPipelineBarrier)(VkCommandBuffer, VkFlags, VkFlags, VkFlags, uint32_t, const VkMemoryBarrier *, uint32_t,
                                    const void *, uint32_t, const void *);
    VkResult(VKAPI *QueueSubmit)(VkQueue, uint32_t, const VkSubmitInfo *, VkFence);
    VkResult(VKAPI *CreateFence)(VkDevice, const VkFenceCreateInfo *, const void *, VkFence *);
    void(VKAPI *DestroyFence)(VkDevice, VkFence, const void *);
    VkResult(VKAPI *ResetFences)(VkDevice, uint32_t, const VkFence *);
    VkResult(VKAPI *GetFenceStatus)(VkDevice, VkFence);
    VkResult(VKAPI *WaitForFences)(VkDevice, uint32_t, const VkFence *, VkBool32, uint64_t);
    VkResult(VKAPI *DeviceWaitIdle)(VkDevice);
} VkApi;

/* the loader and the instance-level entry points; 0 if Vulkan is not installed */
static int vk_load(VkApi *vk) {
    static const char *names[] = {
#if defined(_WIN32)
        "vulkan-1.dll",
#elif defined(__APPLE__)
        "libvulkan.1.dylib", "libMoltenVK.dylib",
#else
        "libvulkan.so.1", "libvulkan.so",
#endif
    };
    memset(vk, 0, sizeof(*vk));
    void *lib = NULL;
    for (size_t i = 0; i < sizeof(names) / sizeof(names[0]) && !lib; i++) lib = lib_open(names[i]);
    if (!lib || !(*(void **)&vk->GetInstanceProcAddr = lib_sym(lib, "vkGetInstanceProcAddr"))) return 0;
    *(void **)&vk->CreateInstance = (void *)vk->GetInstanceProcAddr(NULL, "vkCreateInstance");
    return vk->CreateInstance != NULL;
}

static int vk_load_instance(VkApi *vk, VkInstance inst) {
#define LOAD(f) if (!(*(void **)&vk->f = (void *)vk->GetInstanceProcAddr(inst, "vk" #f))) return 0
    LOAD(DestroyInstance); LOAD(EnumeratePhysicalDevices); LOAD(GetPhysicalDeviceProperties);
    LOAD(GetPhysicalDeviceMemoryProperties); LOAD(GetPhysicalDeviceQueueFamilyProperties); LOAD(CreateDevice);
    LOAD(GetDeviceProcAddr);
#undef LOAD
    return 1;
}

static int vk_load_device(VkApi *vk, VkDevice dev) {
#define LOAD(f) if (!(*(void **)&vk->f = (void *)vk->GetDeviceProcAddr(dev, "vk" #f))) return 0
    LOAD(DestroyDevice); LOAD(GetDeviceQueue); LOAD(CreateBuffer); LOAD(DestroyBuffer); LOAD(GetBufferMemoryRequirements);
    LOAD(AllocateMemory); LOAD(FreeMemory); LOAD(BindBufferMemory); LOAD(MapMemory); LOAD(CreateShaderModule);
    LOAD(DestroyShaderModule); LOAD(CreateDescriptorSetLayout); LOAD(DestroyDescriptorSetLayout); LOAD(CreatePipelineLayout);
    LOAD(DestroyPipelineLayout); LOAD(CreateComputePipelines); LOAD(DestroyPipeline); LOAD(CreateDescriptorPool);
    LOAD(DestroyDescriptorPool); LOAD(ResetDescriptorPool); LOAD(AllocateDescriptorSets); LOAD(UpdateDescriptorSets);
    LOAD(CreateCommandPool); LOAD(DestroyCommandPool); LOAD(ResetCommandPool); LOAD(AllocateCommandBuffers);
    LOAD(BeginCommandBuffer); LOAD(EndCommandBuffer); LOAD(CmdBindPipeline); LOAD(CmdBindDescriptorSets); LOAD(CmdDispatch);
    LOAD(CmdCopyBuffer); LOAD(CmdPipelineBarrier); LOAD(QueueSubmit); LOAD(CreateFence); LOAD(DestroyFence);
    LOAD(ResetFences); LOAD(GetFenceStatus); LOAD(WaitForFences); LOAD(DeviceWaitIdle);
#undef LOAD
    return 1;
}

static VkInstance vk_instance(VkApi *vk) {
    VkApplicationInfo app = {ST_APPLICATION_INFO, NULL, "pithos", 1, "pithos", 1, VK_API_1_1};
    VkInstanceCreateInfo ci = {ST_INSTANCE_CREATE_INFO, NULL, 0, &app, 0, NULL, 0, NULL};
    VkInstance inst = NULL;
    if (vk->CreateInstance(&ci, NULL, &inst) != VK_SUCCESS) return NULL;
    if (!vk_load_instance(vk, inst)) {
        if (vk->DestroyInstance) vk->DestroyInstance(inst, NULL);
        return NULL;
    }
    return inst;
}

/* ------------------------------------------------------------------ state */

/* GPU layouts, 16-byte aligned so that the shaders can read uint4 words (tools/vk_shaders.cmd
 * passes the same numbers to kernels_dx.hlsl): Q8_K activations as int8 qs[256] then float d,
 * padded to 272 bytes; q6_K as ql[128] qh[64] scales[16] then half d, padded to 224 bytes; q8_0
 * in groups of 8 blocks, their int8 qs[256] then the 8 half scales (272 bytes) */
#define Q8_BYTES 272
#define Q8_GPU_D 256
#define Q6K_GPU_BYTES 224
#define Q80_GPU_BYTES 272
#define STAGE_BYTES (64u << 20)
#define VK_MV_TOKENS 12     /* fewer tokens: one matrix-vector product per token */
#define CONST_SLOT 256      /* constants of one dispatch (16 uints, padded to the largest offset alignment) */
#define MAX_SETS 2048       /* dispatches per submit */
#define MAX_DEVICES 16

enum { K_Q4_K, K_Q6_K, K_Q8_0, N_KERNELS };
enum { T_RESID, T_QUANT, T_SILU, T_CONV, T_PREP, T_ATTN, T_CONV_CHUNK, T_CONV_STATE, T_ATTN_CHUNK,
       T_MOE_RESID, T_DCONV, T_DREC, T_QPREP, T_QATTN, T_ROUTER, T_COMBINE, N_TOKEN_KERNELS };   /* kernels_dx_token.hlsl */
enum { B_U0 = 0, B_U1 = 1, B_U2 = 2, B_T0 = 8, B_T1 = 9, B_CONST = 16 };   /* descriptor bindings */

typedef struct {
    VkBuffer buf;
    VkDeviceMemory mem;
    VkDeviceSize size;
    uint8_t *map;                   /* host-visible buffers stay mapped */
} VBuf;

#define MAX_PARTS 8         /* buffers of one matrix, when it is over the device's buffer size limit */

typedef struct {
    const uint8_t *host;
    int kernel, rows, K;
    size_t row_bytes, gpu_row;      /* bytes per row in the file / on the GPU */
    int nparts, part_rows;          /* rows [p * part_rows, ...) in part[p]; nparts 0: not on the GPU */
    VBuf part[MAX_PARTS];
} VMat;

enum { HOST_NONE, HOST_UPLOAD, HOST_READBACK };

#define SET_CACHE 2048      /* slots of the descriptor set cache (half of them usable) */
typedef struct { VkBuffer buf[5]; VkDeviceSize off[5]; VkDescriptorSet set; } SetEntry;

/* A command buffer with what one submit of it needs. Lane 0 runs the passes; lane 1 the routed
 * experts in VRAM, kicked while lane 0 is recorded (the queue keeps the order of the submits). */
typedef struct {
    VkCommandPool pool;
    VkCommandBuffer cb;
    VkFence fence;
    VkDescriptorPool dpool;         /* the sets of one submit (reset after it) */
    int recording, nsets, inflight; /* inflight: kicked, not waited for */
} Lane;

struct Vulkan {
    VkApi vk;
    VkInstance inst;
    VkPhysicalDevice pd;
    VkPhysicalDeviceProperties props;
    VkPhysicalDeviceMemoryProperties mem;
    VkDevice dev;
    VkQueue queue;
    uint32_t family;
    Lane lanes[2], *ln;             /* the command buffers; ln: the one being recorded */
    VkDescriptorSetLayout dsl;
    VkPipelineLayout layout;
    VkDescriptorPool cpool;         /* the sets of the cache */
    SetEntry sets[SET_CACHE];
    int nsets_cached;
    VkPipeline pso[N_KERNELS], pso_mv[N_KERNELS], pso_tok[N_TOKEN_KERNELS];
    VkPipeline pso_mvx[N_KERNELS];  /* pso_mv for the routed experts in VRAM (MV_EXPERTS) */
    int mv_rows;                    /* rows per group of the pso_mv kernels: 8, or 16 with the Intel build */
    int broken;
    VBuf consts, dummy;             /* the constants rings (MAX_SETS slots per lane); a stand-in for unused bindings */
    VMat *mats;                     /* sorted by host address */
    int nmats, n_big, n_split;      /* matrices too large for the GPU / split into parts */
    VBuf up, x, o, rb;              /* upload, activations, results, readback */
    struct VDense *dense;           /* whole forward pass, or NULL */
};

/* ------------------------------------------------------------------ device listing */

static const char *type_name(int t) {
    return t == 1 ? "integrated" : t == 2 ? "discrete" : t == 3 ? "virtual" : t == 4 ? "CPU" : "other";
}

static VkDeviceSize device_memory(VkApi *vk, VkPhysicalDevice pd) {
    VkPhysicalDeviceMemoryProperties mp;
    vk->GetPhysicalDeviceMemoryProperties(pd, &mp);
    VkDeviceSize best = 0;
    for (uint32_t i = 0; i < mp.memoryHeapCount; i++)
        if ((mp.memoryHeaps[i].flags & HEAP_DEVICE_LOCAL) && mp.memoryHeaps[i].size > best) best = mp.memoryHeaps[i].size;
    return best;
}

static int enum_devices(VkApi *vk, VkInstance inst, VkPhysicalDevice *pds) {
    uint32_t n = MAX_DEVICES;
    const VkResult r = vk->EnumeratePhysicalDevices(inst, &n, pds);
    return r == VK_SUCCESS || r == 5 /* VK_INCOMPLETE */ ? (int)n : 0;
}

void vk_list_devices(void) {
    VkApi vk;
    VkInstance inst = vk_load(&vk) ? vk_instance(&vk) : NULL;
    if (!inst) { printf("  Vulkan not available\n"); return; }
    VkPhysicalDevice pds[MAX_DEVICES];
    const int n = enum_devices(&vk, inst, pds);
    for (int i = 0; i < n; i++) {
        VkPhysicalDeviceProperties p;
        vk.GetPhysicalDeviceProperties(pds[i], &p);
        printf("  [%d] %s | %llu MB | %s | Vulkan %u.%u.%u | buffers up to %u MB, offsets %u\n", i, p.deviceName,
               (unsigned long long)(device_memory(&vk, pds[i]) >> 20), type_name(p.deviceType), p.apiVersion >> 22,
               (p.apiVersion >> 12) & 1023, p.apiVersion & 4095, p.limits.maxStorageBufferRange >> 20,
               (unsigned)p.limits.minStorageBufferOffsetAlignment);
    }
    if (!n) printf("  no Vulkan devices\n");
    vk.DestroyInstance(inst, NULL);
}

/* ------------------------------------------------------------------ buffers and commands */

static int find_memory(const Vulkan *v, uint32_t bits, VkFlags want, VkFlags prefer) {
    for (int pass = 0; pass < 2; pass++)
        for (uint32_t i = 0; i < v->mem.memoryTypeCount; i++) {
            const VkFlags f = v->mem.memoryTypes[i].propertyFlags;
            if ((bits >> i & 1) && (f & want) == want && (pass || (f & prefer) == prefer)) return (int)i;
        }
    return -1;
}

static void vbuf_free(Vulkan *v, VBuf *b) {
    if (b->buf) v->vk.DestroyBuffer(v->dev, b->buf, NULL);
    if (b->mem) v->vk.FreeMemory(v->dev, b->mem, NULL);   /* unmaps it */
    memset(b, 0, sizeof(*b));
}

/* a buffer usable as storage and copy source / destination; host: HOST_* */
static int vbuf_create(Vulkan *v, VBuf *b, VkDeviceSize size, int host) {
    memset(b, 0, sizeof(*b));
    VkBufferCreateInfo bi = {ST_BUFFER_CREATE_INFO, NULL, 0, size,
                             USAGE_STORAGE | USAGE_UNIFORM | USAGE_TRANSFER_SRC | USAGE_TRANSFER_DST, 0, 0, NULL};
    if (v->vk.CreateBuffer(v->dev, &bi, NULL, &b->buf) != VK_SUCCESS) { b->buf = 0; return 0; }
    VkMemoryRequirements req;
    v->vk.GetBufferMemoryRequirements(v->dev, b->buf, &req);
    const int t = host == HOST_NONE ? find_memory(v, req.memoryTypeBits, MEM_DEVICE_LOCAL, 0)
                : host == HOST_UPLOAD ? find_memory(v, req.memoryTypeBits, MEM_HOST_VISIBLE | MEM_HOST_COHERENT, 0)
                : find_memory(v, req.memoryTypeBits, MEM_HOST_VISIBLE | MEM_HOST_COHERENT, MEM_HOST_CACHED);
    VkMemoryAllocateInfo ai = {ST_MEMORY_ALLOCATE_INFO, NULL, req.size, (uint32_t)t};
    void *p = NULL;
    if (t < 0 || v->vk.AllocateMemory(v->dev, &ai, NULL, &b->mem) != VK_SUCCESS) {
        b->mem = 0;
        vbuf_free(v, b);
        return 0;
    }
    if (v->vk.BindBufferMemory(v->dev, b->buf, b->mem, 0) != VK_SUCCESS ||
        (host != HOST_NONE && v->vk.MapMemory(v->dev, b->mem, 0, VK_WHOLE_SIZE, 0, &p) != VK_SUCCESS)) {
        vbuf_free(v, b);
        return 0;
    }
    b->size = size;
    b->map = (uint8_t *)p;
    return 1;
}

/* (re)create b with at least `size` bytes */
static int vbuf_reserve(Vulkan *v, VBuf *b, VkDeviceSize size, int host) {
    if (b->buf && b->size >= size) return 1;
    vbuf_free(v, b);
    return vbuf_create(v, b, (size + 0xFFFFF) & ~(VkDeviceSize)0xFFFFF, host);
}

static int wait_gpu(Vulkan *v);

static VkCommandBuffer cmd(Vulkan *v) {
    if (v->ln->inflight) wait_gpu(v);   /* its buffer and sets are about to be reused */
    if (!v->ln->recording) {
        VkCommandBufferBeginInfo bi = {ST_COMMAND_BUFFER_BEGIN_INFO, NULL, CB_ONE_TIME, NULL};
        v->vk.BeginCommandBuffer(v->ln->cb, &bi);
        v->ln->recording = 1;
        /* what earlier submits wrote (KV cache, state, copies) is visible to this one */
        VkMemoryBarrier mb = {ST_MEMORY_BARRIER, NULL, AC_MEMORY_WRITE, AC_MEMORY_READ | AC_MEMORY_WRITE};
        v->vk.CmdPipelineBarrier(v->ln->cb, PS_ALL, PS_ALL, 0, 1, &mb, 0, NULL, 0, NULL);
    }
    return v->ln->cb;
}

/* one global memory barrier */
static void barrier(Vulkan *v, VkFlags src_stage, VkFlags src_access, VkFlags dst_stage, VkFlags dst_access) {
    VkMemoryBarrier mb = {ST_MEMORY_BARRIER, NULL, src_access, dst_access};
    v->vk.CmdPipelineBarrier(cmd(v), src_stage, dst_stage, 0, 1, &mb, 0, NULL, 0, NULL);
}

static void copy(Vulkan *v, const VBuf *dst, VkDeviceSize doff, const VBuf *src, VkDeviceSize soff, VkDeviceSize bytes) {
    VkBufferCopy r = {soff, doff, bytes};
    v->vk.CmdCopyBuffer(cmd(v), src->buf, dst->buf, 1, &r);
}

/* end the command buffer and start it on the GPU; wait_gpu waits for it */
static int kick(Vulkan *v) {
    if (!v->ln->recording) return 1;
    v->ln->recording = 0;
    if (v->vk.EndCommandBuffer(v->ln->cb) != VK_SUCCESS) {
        fprintf(stderr, "Vulkan: invalid command buffer\n");
        return 0;
    }
    VkSubmitInfo si = {ST_SUBMIT_INFO, NULL, 0, NULL, NULL, 1, &v->ln->cb, 0, NULL};
    const VkResult r = v->vk.QueueSubmit(v->queue, 1, &si, v->ln->fence);
    if (r != VK_SUCCESS) {
        fprintf(stderr, "Vulkan: the GPU did not complete the work (error %d)\n", (int)r);
        return 0;
    }
    v->ln->inflight = 1;
    return 1;
}

/* wait for the kicked command buffer, and get ready for the next one */
static int wait_gpu(Vulkan *v) {
    if (!v->ln->inflight) return 1;
    v->ln->inflight = 0;
    /* poll for up to 50 ms (PITHOS_VK_SPIN, in ms): waking up from vkWaitForFences can take
     * about a millisecond, 8% of a token on a GTX 1650 */
    const double t0 = time_sec(), spin = getenv("PITHOS_VK_SPIN") ? atof(getenv("PITHOS_VK_SPIN")) * 1e-3 : 50e-3;
    VkResult r;
    while ((r = v->vk.GetFenceStatus(v->dev, v->ln->fence)) == VK_NOT_READY && time_sec() - t0 < spin) cpu_relax();
    if (r == VK_NOT_READY) r = v->vk.WaitForFences(v->dev, 1, &v->ln->fence, 1, 60000000000ull);
    if (r != VK_SUCCESS) {
        fprintf(stderr, "Vulkan: the GPU did not complete the work (error %d)\n", (int)r);
        return 0;
    }
    v->ln->nsets = 0;
    return v->vk.ResetFences(v->dev, 1, &v->ln->fence) == VK_SUCCESS &&
           v->vk.ResetCommandPool(v->dev, v->ln->pool, 0) == VK_SUCCESS &&
           v->vk.ResetDescriptorPool(v->dev, v->ln->dpool, 0) == VK_SUCCESS;
}

/* end the command buffer, run it, wait, and get ready for the next one */
static int submit(Vulkan *v) {
    return kick(v) && wait_gpu(v);
}

typedef struct { const VBuf *b; VkDeviceSize off; } Bind;

/* a descriptor set for these five bindings (the constants slot comes as a dynamic offset) */
static VkDescriptorSet make_set(Vulkan *v, VkDescriptorPool pool, const Bind b[5]) {
    VkDescriptorSet set = 0;
    VkDescriptorSetAllocateInfo ai = {ST_DESCRIPTOR_SET_ALLOCATE_INFO, NULL, pool, 1, &v->dsl};
    if (v->vk.AllocateDescriptorSets(v->dev, &ai, &set) != VK_SUCCESS) return 0;
    const uint32_t slots[6] = {B_U0, B_U1, B_U2, B_T0, B_T1, B_CONST};
    VkDescriptorBufferInfo info[6];
    VkWriteDescriptorSet w[6];
    for (int i = 0; i < 6; i++) {
        const int cst = i == 5;
        info[i].buffer = cst ? v->consts.buf : b[i].b ? b[i].b->buf : v->dummy.buf;
        info[i].offset = !cst && b[i].b ? b[i].off : 0;
        info[i].range = cst ? CONST_SLOT : VK_WHOLE_SIZE;
        VkWriteDescriptorSet x = {ST_WRITE_DESCRIPTOR_SET, NULL, set, slots[i], 0, 1,
                                  cst ? DESC_UNIFORM_DYNAMIC : DESC_STORAGE_BUFFER, NULL, &info[i], NULL};
        w[i] = x;
    }
    v->vk.UpdateDescriptorSets(v->dev, 6, w, 0, NULL);
    return set;
}

/* the set for b from the cache of sets over long-lived buffers (the whole forward pass binds the
 * same few hundred combinations every token); 0 when the cache is full */
static VkDescriptorSet cached_set(Vulkan *v, const Bind b[5]) {
    uint64_t h = 1469598103934665603ull;
    for (int i = 0; i < 5; i++) {
        h = (h ^ (b[i].b ? b[i].b->buf : 0)) * 1099511628211ull;
        h = (h ^ b[i].off) * 1099511628211ull;
    }
    for (uint32_t k = 0, i = (uint32_t)(h % SET_CACHE); k < SET_CACHE; k++, i = (i + 1) % SET_CACHE) {
        SetEntry *e = &v->sets[i];
        if (!e->set) {
            if (v->nsets_cached == SET_CACHE / 2) return 0;   /* keep probing short */
            e->set = make_set(v, v->cpool, b);
            if (!e->set) return 0;
            v->nsets_cached++;
            for (int j = 0; j < 5; j++) {
                e->buf[j] = b[j].b ? b[j].b->buf : 0;
                e->off[j] = b[j].off;
            }
            return e->set;
        }
        int same = 1;
        for (int j = 0; j < 5 && same; j++) same = e->buf[j] == (b[j].b ? b[j].b->buf : 0) && e->off[j] == b[j].off;
        if (same) return e->set;
    }
    return 0;
}

/* record one dispatch of pipeline p with constants c[nc] and the buffers of the D3D12 root
 * parameters (NULL buffer: unused); cached: the buffers live as long as the device (their set
 * is kept). 0 if a descriptor set could not be had. */
static int dispatch(Vulkan *v, VkPipeline p, const uint32_t *c, int nc, Bind u0, Bind u1, Bind u2, Bind t0, Bind t1,
                    uint32_t gx, uint32_t gy, int cached) {
    if (v->ln->nsets == MAX_SETS && !submit(v)) return 0;   /* rare: flush and continue in a new command buffer */
    const Bind b[5] = {u0, u1, u2, t0, t1};
    VkDescriptorSet set = cached ? cached_set(v, b) : 0;
    if (!set) set = make_set(v, v->ln->dpool, b);
    if (!set) return 0;
    const uint32_t coff = (uint32_t)((v->ln - v->lanes) * MAX_SETS + v->ln->nsets) * CONST_SLOT;
    memset(v->consts.map + coff, 0, CONST_SLOT);
    memcpy(v->consts.map + coff, c, sizeof(uint32_t) * (size_t)nc);
    v->ln->nsets++;
    VkCommandBuffer cb = cmd(v);
    v->vk.CmdBindPipeline(cb, BIND_COMPUTE, p);
    v->vk.CmdBindDescriptorSets(cb, BIND_COMPUTE, v->layout, 0, 1, &set, 1, &coff);
    v->vk.CmdDispatch(cb, gx, gy, 1);
    return 1;
}

/* ------------------------------------------------------------------ setup */

static int create_pipelines(Vulkan *v) {
    VkDescriptorSetLayoutBinding b[6];
    const uint32_t slots[6] = {B_U0, B_U1, B_U2, B_T0, B_T1, B_CONST};
    for (int i = 0; i < 6; i++) {
        VkDescriptorSetLayoutBinding x = {slots[i], i == 5 ? DESC_UNIFORM_DYNAMIC : DESC_STORAGE_BUFFER, 1,
                                          STAGE_COMPUTE_SHADER_BIT, NULL};
        b[i] = x;
    }
    VkDescriptorSetLayoutCreateInfo li = {ST_DESCRIPTOR_SET_LAYOUT_CREATE_INFO, NULL, 0, 6, b};
    if (v->vk.CreateDescriptorSetLayout(v->dev, &li, NULL, &v->dsl) != VK_SUCCESS) return 0;
    VkPipelineLayoutCreateInfo pi = {ST_PIPELINE_LAYOUT_CREATE_INFO, NULL, 0, 1, &v->dsl, 0, NULL};
    if (v->vk.CreatePipelineLayout(v->dev, &pi, NULL, &v->layout) != VK_SUCCESS) return 0;
    VkDescriptorPoolSize ps[2] = {{DESC_STORAGE_BUFFER, 5 * MAX_SETS}, {DESC_UNIFORM_DYNAMIC, MAX_SETS}};
    VkDescriptorPoolCreateInfo dpi = {ST_DESCRIPTOR_POOL_CREATE_INFO, NULL, 0, MAX_SETS, 2, ps};
    for (int i = 0; i < 2; i++)
        if (v->vk.CreateDescriptorPool(v->dev, &dpi, NULL, &v->lanes[i].dpool) != VK_SUCCESS) return 0;
    VkDescriptorPoolSize cps[2] = {{DESC_STORAGE_BUFFER, 5 * SET_CACHE / 2}, {DESC_UNIFORM_DYNAMIC, SET_CACHE / 2}};
    VkDescriptorPoolCreateInfo cpi = {ST_DESCRIPTOR_POOL_CREATE_INFO, NULL, 0, SET_CACHE / 2, 2, cps};
    if (v->vk.CreateDescriptorPool(v->dev, &cpi, NULL, &v->cpool) != VK_SUCCESS) return 0;

    static const struct { const unsigned char *code; size_t size; const char *entry; } k[3 * N_KERNELS + N_TOKEN_KERNELS] = {
        {g_vk_mm_q4_K, sizeof(g_vk_mm_q4_K), "mm_q4_K"}, {g_vk_mm_q6_K, sizeof(g_vk_mm_q6_K), "mm_q6_K"},
        {g_vk_mm_q8_0, sizeof(g_vk_mm_q8_0), "mm_q8_0"},
        {g_vk_mv_q4_K, sizeof(g_vk_mv_q4_K), "mv_q4_K"}, {g_vk_mv_q6_K, sizeof(g_vk_mv_q6_K), "mv_q6_K"},
        {g_vk_mv_q8_0, sizeof(g_vk_mv_q8_0), "mv_q8_0"},
        {g_vk_resid_norm, sizeof(g_vk_resid_norm), "resid_norm"}, {g_vk_quant, sizeof(g_vk_quant), "quant"},
        {g_vk_silu_quant, sizeof(g_vk_silu_quant), "silu_quant"}, {g_vk_shortconv, sizeof(g_vk_shortconv), "shortconv"},
        {g_vk_attn_prep, sizeof(g_vk_attn_prep), "attn_prep"}, {g_vk_attention, sizeof(g_vk_attention), "attention"},
        {g_vk_conv_chunk, sizeof(g_vk_conv_chunk), "conv_chunk"}, {g_vk_conv_state, sizeof(g_vk_conv_state), "conv_state"},
        {g_vk_attn_chunk, sizeof(g_vk_attn_chunk), "attn_chunk"},
        {g_vk_moe_resid, sizeof(g_vk_moe_resid), "moe_resid"}, {g_vk_delta_conv, sizeof(g_vk_delta_conv), "delta_conv"},
        {g_vk_delta_rec, sizeof(g_vk_delta_rec), "delta_rec"}, {g_vk_qattn_prep, sizeof(g_vk_qattn_prep), "qattn_prep"},
        {g_vk_qattn, sizeof(g_vk_qattn), "qattn"}, {g_vk_router, sizeof(g_vk_router), "router"},
        {g_vk_moe_combine, sizeof(g_vk_moe_combine), "moe_combine"},
        {g_vk_mvx_q4_K, sizeof(g_vk_mvx_q4_K), "mv_q4_K"}, {g_vk_mvx_q6_K, sizeof(g_vk_mvx_q6_K), "mv_q6_K"},
        {g_vk_mvx_q8_0, sizeof(g_vk_mvx_q8_0), "mv_q8_0"},
    };
    /* Intel: the matmul kernels built for its GPUs (kernels_dx.hlsl); PITHOS_VK_INTEL=0: the common ones */
    static const struct { const unsigned char *code; size_t size; } intel_k[2 * N_KERNELS] = {
        {g_vk_mm_q4_K_intel, sizeof(g_vk_mm_q4_K_intel)}, {g_vk_mm_q6_K_intel, sizeof(g_vk_mm_q6_K_intel)},
        {g_vk_mm_q8_0_intel, sizeof(g_vk_mm_q8_0_intel)},
        {g_vk_mv_q4_K_intel, sizeof(g_vk_mv_q4_K_intel)}, {g_vk_mv_q6_K_intel, sizeof(g_vk_mv_q6_K_intel)},
        {g_vk_mv_q8_0_intel, sizeof(g_vk_mv_q8_0_intel)},
    };
    const char *env = getenv("PITHOS_VK_INTEL");
    const int intel = v->props.vendorID == 0x8086 && !(env && env[0] == '0');
    v->mv_rows = intel ? 16 : 8;
    for (int i = 0; i < 3 * N_KERNELS + N_TOKEN_KERNELS; i++) {
        const int own = intel && i < 2 * N_KERNELS;
        const unsigned char *code = own ? intel_k[i].code : k[i].code;
        const size_t size = own ? intel_k[i].size : k[i].size;
        uint32_t *words = (uint32_t *)malloc(size);   /* the module wants 4-byte aligned code */
        memcpy(words, code, size);
        VkShaderModuleCreateInfo si = {ST_SHADER_MODULE_CREATE_INFO, NULL, 0, size, words};
        VkShaderModule mod = 0;
        VkResult r = v->vk.CreateShaderModule(v->dev, &si, NULL, &mod);
        free(words);
        if (r != VK_SUCCESS) return 0;
        VkComputePipelineCreateInfo ci = {ST_COMPUTE_PIPELINE_CREATE_INFO, NULL, 0,
                                          {ST_PIPELINE_SHADER_STAGE_CREATE_INFO, NULL, 0, STAGE_COMPUTE_SHADER_BIT, mod,
                                           k[i].entry, NULL},
                                          v->layout, 0, -1};
        VkPipeline *dst = i < N_KERNELS ? &v->pso[i] : i < 2 * N_KERNELS ? &v->pso_mv[i - N_KERNELS]
                        : i < 2 * N_KERNELS + N_TOKEN_KERNELS ? &v->pso_tok[i - 2 * N_KERNELS]
                                                              : &v->pso_mvx[i - 2 * N_KERNELS - N_TOKEN_KERNELS];
        r = v->vk.CreateComputePipelines(v->dev, 0, 1, &ci, NULL, dst);
        v->vk.DestroyShaderModule(v->dev, mod, NULL);
        if (r != VK_SUCCESS) return 0;
    }
    return vbuf_create(v, &v->consts, (VkDeviceSize)2 * MAX_SETS * CONST_SLOT, HOST_UPLOAD) &&
           vbuf_create(v, &v->dummy, 256, HOST_NONE);
}

/* x describes w (at host), with no GPU buffers yet; 0 if no kernel takes its type */
static int vmat_init(VMat *x, const Mat *w, const uint8_t *host) {
    const int kernel = w->type == GGML_Q4_K ? K_Q4_K : w->type == GGML_Q6_K ? K_Q6_K : w->type == GGML_Q8_0 ? K_Q8_0 : -1;
    if (!host || kernel < 0 || w->cols % QK_K) return 0;
    memset(x, 0, sizeof(*x));
    x->host = host;
    x->kernel = kernel;
    x->rows = w->rows;
    x->K = w->cols;
    x->row_bytes = w->row_bytes;
    x->gpu_row = kernel == K_Q6_K ? (size_t)(w->cols / QK_K) * Q6K_GPU_BYTES
               : kernel == K_Q8_0 ? (size_t)(w->cols / QK_K) * Q80_GPU_BYTES : w->row_bytes;
    return 1;
}

static void add_mat(Vulkan *v, const Mat *w, int *cap) {
    VMat x;
    if (!vmat_init(&x, w, w->data)) return;
    if (v->nmats == *cap) {
        *cap = *cap ? *cap * 2 : 64;
        v->mats = (VMat *)realloc(v->mats, sizeof(VMat) * (size_t)*cap);
    }
    v->mats[v->nmats++] = x;
}

static int mat_cmp(const void *a, const void *b) {
    const uint8_t *x = ((const VMat *)a)->host, *y = ((const VMat *)b)->host;
    return x < y ? -1 : x > y;
}

static const VMat *find_mat(const Vulkan *v, const uint8_t *w) {
    int lo = 0, hi = v->nmats - 1;
    while (lo <= hi) {
        const int mid = (lo + hi) / 2;
        if (v->mats[mid].host == w) return &v->mats[mid];
        if (v->mats[mid].host < w) lo = mid + 1;
        else hi = mid - 1;
    }
    return NULL;
}

/* file layout -> GPU layout of rows [r0, r1) of a matrix */
static void pack_rows(const VMat *x, int r0, int r1, uint8_t *dst) {
    const uint8_t *src = x->host + (size_t)r0 * x->row_bytes;
    const size_t nblk = (size_t)(r1 - r0) * (size_t)(x->K / QK_K);
    if (x->kernel == K_Q8_0) {
        const block_q8_0 *b = (const block_q8_0 *)src;
        for (size_t i = 0; i < nblk; i++, dst += Q80_GPU_BYTES)
            for (int j = 0; j < QK_K / QK8_0; j++) {
                memcpy(dst + j * QK8_0, b[i * (QK_K / QK8_0) + j].qs, QK8_0);
                memcpy(dst + QK_K + j * 2, &b[i * (QK_K / QK8_0) + j].d, 2);
            }
        return;
    }
    if (x->kernel != K_Q6_K) {
        memcpy(dst, src, x->row_bytes * (size_t)(r1 - r0));
        return;
    }
    const block_q6_K *b = (const block_q6_K *)src;
    for (size_t i = 0; i < nblk; i++, dst += Q6K_GPU_BYTES) {
        memcpy(dst, b[i].ql, 128);
        memcpy(dst + 128, b[i].qh, 64);
        memcpy(dst + 192, b[i].scales, 16);
        memset(dst + 208, 0, Q6K_GPU_BYTES - 208);
        memcpy(dst + 208, &b[i].d, 2);
    }
}

/* matrices go up through the staging buffer, in pieces of whole rows when they do not fit */
static int upload_weights(Vulkan *v, char *err, int errlen) {
    if (!vbuf_reserve(v, &v->up, STAGE_BYTES, HOST_UPLOAD)) {
        snprintf(err, (size_t)errlen, "cannot allocate the upload buffer");
        return 0;
    }
    const VkDeviceSize max_buf = v->props.limits.maxStorageBufferRange;
    size_t used = 0;
    for (int i = 0; i < v->nmats; i++) {
        VMat *x = &v->mats[i];
        /* over the limit of one buffer: parts of whole 64-row tiles (their outputs, 256 bytes apart,
         * stay bindable) */
        x->part_rows = x->rows;
        if ((VkDeviceSize)x->gpu_row * (VkDeviceSize)x->rows > max_buf) {
            x->part_rows = (int)(max_buf / x->gpu_row) & ~63;
            if (x->part_rows < 64 || (x->rows + x->part_rows - 1) / x->part_rows > MAX_PARTS) {
                v->n_big++;   /* stays on the CPU */
                continue;
            }
            v->n_split++;
        }
        for (int p = 0; p * x->part_rows < x->rows; p++) {
            const int n = x->rows - p * x->part_rows < x->part_rows ? x->rows - p * x->part_rows : x->part_rows;
            if (!vbuf_create(v, &x->part[p], (VkDeviceSize)x->gpu_row * (VkDeviceSize)n, HOST_NONE)) {
                snprintf(err, (size_t)errlen, "out of GPU memory after %d of %d matrices", i, v->nmats);
                return 0;
            }
            x->nparts = p + 1;
        }
        for (int r0 = 0; r0 < x->rows;) {
            int n = (int)((v->up.size - used) / x->gpu_row);
            const int p = r0 / x->part_rows, end = p * x->part_rows + x->part_rows < x->rows
                                                     ? p * x->part_rows + x->part_rows : x->rows;
            if (n > end - r0) n = end - r0;
            if (n < 1 || (n < end - r0 && used)) {   /* full: send what is staged */
                if (!submit(v)) { snprintf(err, (size_t)errlen, "weight upload failed"); return 0; }
                used = 0;
                continue;
            }
            const size_t bytes = x->gpu_row * (size_t)n;
            pack_rows(x, r0, r0 + n, v->up.map + used);
            copy(v, &x->part[p], (VkDeviceSize)x->gpu_row * (VkDeviceSize)(r0 - p * x->part_rows), &v->up, used,
                 bytes);
            used = (used + bytes + 255) & ~(size_t)255;
            r0 += n;
        }
    }
    if (!submit(v)) { snprintf(err, (size_t)errlen, "weight upload failed"); return 0; }
    vbuf_free(v, &v->up);   /* vk_matmul allocates what it needs */
    return 1;
}

/* the requested device, or the discrete GPU with the most memory (else any GPU), never a CPU
 * implementation unless asked for by index; VK_AUTO_DEVICE: discrete NVIDIA GPUs only, and not
 * through WSL's Direct3D 12 layer (dozen, "Microsoft Direct3D12 (...)") */
static int pick_device(Vulkan *v, int want, char *info, int infolen) {
    VkPhysicalDevice pds[MAX_DEVICES];
    const int n = enum_devices(&v->vk, v->inst, pds);
    if (!n) { snprintf(info, (size_t)infolen, "no Vulkan devices"); return 0; }
    if (want >= n) { snprintf(info, (size_t)infolen, "no Vulkan device %d (there are %d)", want, n); return 0; }
    int best = want;
    long long best_score = -1;
    for (int i = 0; i < n && want < 0; i++) {
        VkPhysicalDeviceProperties p;
        v->vk.GetPhysicalDeviceProperties(pds[i], &p);
        if (p.deviceType == 4 || p.apiVersion < VK_API_1_1) continue;
        if (want == VK_AUTO_DEVICE && (p.deviceType != 2 || p.vendorID != 0x10DE || !strncmp(p.deviceName, "Microsoft", 9)))
            continue;
        const long long score = (p.deviceType == 2 ? 1ll << 50 : 0) + (long long)(device_memory(&v->vk, pds[i]) >> 20);
        if (score > best_score) { best_score = score; best = i; }
    }
    if (best < 0) {
        snprintf(info, (size_t)infolen, "%s", want == VK_AUTO_DEVICE ? "no discrete NVIDIA GPU"
                                              : "no Vulkan 1.1 GPU (a CPU implementation needs its device index)");
        return 0;
    }
    v->pd = pds[best];
    v->vk.GetPhysicalDeviceProperties(v->pd, &v->props);
    v->vk.GetPhysicalDeviceMemoryProperties(v->pd, &v->mem);
    if (v->props.apiVersion < VK_API_1_1) {
        snprintf(info, (size_t)infolen, "%s supports only Vulkan %u.%u", v->props.deviceName, v->props.apiVersion >> 22,
                 (v->props.apiVersion >> 12) & 1023);
        return 0;
    }
    VkQueueFamilyProperties qf[16];
    uint32_t nq = 16;
    v->vk.GetPhysicalDeviceQueueFamilyProperties(v->pd, &nq, qf);
    for (v->family = 0; v->family < nq && !(qf[v->family].queueFlags & QUEUE_COMPUTE); v->family++) {}
    if (v->family == nq) { snprintf(info, (size_t)infolen, "%s has no compute queue", v->props.deviceName); return 0; }
    /* PITHOS_VK_MAXBUF=MB: a lower buffer size limit, to test the splitting of large matrices */
    const double lim = getenv("PITHOS_VK_MAXBUF") ? atof(getenv("PITHOS_VK_MAXBUF")) * 1048576.0 : 0;
    if (lim >= 1048576.0 && lim < v->props.limits.maxStorageBufferRange) v->props.limits.maxStorageBufferRange = (uint32_t)lim;
    return 1;
}

static void dense_free(Vulkan *v);

Vulkan *vk_create(const Model *m, int device, char *info, int infolen) {
    info[0] = 0;
    Vulkan *v = (Vulkan *)calloc(1, sizeof(Vulkan));
    if (!vk_load(&v->vk)) {
        snprintf(info, (size_t)infolen, "Vulkan not installed (no vulkan loader)");
        free(v);
        return NULL;
    }
    v->inst = vk_instance(&v->vk);
    if (!v->inst) {
        snprintf(info, (size_t)infolen, "no Vulkan 1.1 instance");
        free(v);
        return NULL;
    }
    if (!pick_device(v, device, info, infolen)) {
        vk_destroy(v);
        return NULL;
    }
    const float prio = 1.f;
    VkDeviceQueueCreateInfo qi = {ST_DEVICE_QUEUE_CREATE_INFO, NULL, 0, v->family, 1, &prio};
    VkDeviceCreateInfo di = {ST_DEVICE_CREATE_INFO, NULL, 0, 1, &qi, 0, NULL, 0, NULL, NULL};
    VkCommandPoolCreateInfo cpi = {ST_COMMAND_POOL_CREATE_INFO, NULL, 0, v->family};
    VkFenceCreateInfo fi = {ST_FENCE_CREATE_INFO, NULL, 0};
    const char *name = v->props.deviceName;
    if (v->vk.CreateDevice(v->pd, &di, NULL, &v->dev) != VK_SUCCESS || !vk_load_device(&v->vk, v->dev)) {
        snprintf(info, (size_t)infolen, "cannot open %s", name);
        if (!v->vk.DestroyDevice) v->dev = NULL;
        vk_destroy(v);
        return NULL;
    }
    v->vk.GetDeviceQueue(v->dev, v->family, 0, &v->queue);
    VkCommandBufferAllocateInfo cai = {ST_COMMAND_BUFFER_ALLOCATE_INFO, NULL, 0, 0, 1};
    int ok = 1;
    v->ln = &v->lanes[0];
    for (int i = 0; i < 2 && ok; i++) {
        Lane *ln = &v->lanes[i];
        ok = v->vk.CreateCommandPool(v->dev, &cpi, NULL, &ln->pool) == VK_SUCCESS;
        cai.commandPool = ln->pool;
        ok = ok && v->vk.AllocateCommandBuffers(v->dev, &cai, &ln->cb) == VK_SUCCESS &&
             v->vk.CreateFence(v->dev, &fi, NULL, &ln->fence) == VK_SUCCESS;
    }
    ok = ok && create_pipelines(v);
    if (!ok) {
        snprintf(info, (size_t)infolen, "Vulkan setup failed on %s", name);
        vk_destroy(v);
        return NULL;
    }

    int cap = 0;
    for (int l = 0; l < m->n_layer; l++) {
        const Layer *L = &m->layers[l];
        const Mat *ws[] = {&L->in_proj, &L->out_proj, &L->wq, &L->wk, &L->wv, &L->wo, &L->attn_gate,
                           &L->w_gate, &L->w_up, &L->w_down, &L->ssm_qkv, &L->ssm_z, &L->ssm_alpha,
                           &L->ssm_beta, &L->ssm_out, &L->sh_gate, &L->sh_up, &L->sh_down};
        for (size_t i = 0; i < sizeof(ws) / sizeof(ws[0]); i++) add_mat(v, ws[i], &cap);
    }
    add_mat(v, &m->output, &cap);
    if (!v->nmats) {
        snprintf(info, (size_t)infolen, "no Q4_K / Q6_K / Q8_0 dense weights in this model");
        vk_destroy(v);
        return NULL;
    }
    qsort(v->mats, (size_t)v->nmats, sizeof(VMat), mat_cmp);
    char err[256];
    if (!upload_weights(v, err, sizeof(err))) {
        snprintf(info, (size_t)infolen, "%s: %s", name, err);
        vk_destroy(v);
        return NULL;
    }
    size_t total = 0;
    for (int i = 0; i < v->nmats; i++)
        if (v->mats[i].nparts) total += v->mats[i].gpu_row * (size_t)v->mats[i].rows;
    char big[128] = "";
    const unsigned lim = (unsigned)(v->props.limits.maxStorageBufferRange >> 20);
    if (v->n_big)
        snprintf(big, sizeof(big), " (%d over the %u MB buffer limit stay on the CPU)", v->n_big, lim);
    else if (v->n_split)
        snprintf(big, sizeof(big), " (%d split at the %u MB buffer limit)", v->n_split, lim);
    snprintf(info, (size_t)infolen, "%s (%s) | %d matrices%s, %.0f MB of weights", name, type_name(v->props.deviceType),
             v->nmats - v->n_big, big, total / 1048576.0);
    return v;
}

void vk_destroy(Vulkan *v) {
    if (!v) return;
    if (v->dev && v->vk.DeviceWaitIdle) {   /* every device entry point was loaded */
        v->vk.DeviceWaitIdle(v->dev);
        for (int i = 0; i < v->nmats; i++)
            for (int p = 0; p < v->mats[i].nparts; p++) vbuf_free(v, &v->mats[i].part[p]);
        dense_free(v);
        VBuf *bufs[] = {&v->up, &v->x, &v->o, &v->rb, &v->consts, &v->dummy};
        for (size_t i = 0; i < sizeof(bufs) / sizeof(bufs[0]); i++) vbuf_free(v, bufs[i]);
        for (int k = 0; k < N_KERNELS; k++) {
            if (v->pso[k]) v->vk.DestroyPipeline(v->dev, v->pso[k], NULL);
            if (v->pso_mv[k]) v->vk.DestroyPipeline(v->dev, v->pso_mv[k], NULL);
            if (v->pso_mvx[k]) v->vk.DestroyPipeline(v->dev, v->pso_mvx[k], NULL);
        }
        for (int k = 0; k < N_TOKEN_KERNELS; k++)
            if (v->pso_tok[k]) v->vk.DestroyPipeline(v->dev, v->pso_tok[k], NULL);
        if (v->cpool) v->vk.DestroyDescriptorPool(v->dev, v->cpool, NULL);
        if (v->layout) v->vk.DestroyPipelineLayout(v->dev, v->layout, NULL);
        if (v->dsl) v->vk.DestroyDescriptorSetLayout(v->dev, v->dsl, NULL);
        for (int i = 0; i < 2; i++) {
            Lane *ln = &v->lanes[i];
            if (ln->dpool) v->vk.DestroyDescriptorPool(v->dev, ln->dpool, NULL);
            if (ln->fence) v->vk.DestroyFence(v->dev, ln->fence, NULL);
            if (ln->pool) v->vk.DestroyCommandPool(v->dev, ln->pool, NULL);   /* frees the command buffer */
        }
        v->vk.DestroyDevice(v->dev, NULL);
    }
    if (v->inst) v->vk.DestroyInstance(v->inst, NULL);
    free(v->mats);
    free(v);
}

/* out[t][r] (rows orow floats apart) = rows of matrix x (as laid out at w) . the Q8_K rows of n
 * tokens, xrow bytes apart */
static int mm_rows(Vulkan *v, const VMat *x, Bind w, int rows, Bind o, Bind in, int n, int orow, uint32_t xrow, int cached) {
    const uint32_t c[6] = {(uint32_t)rows, (uint32_t)(x->K / QK_K), (uint32_t)n, (uint32_t)x->gpu_row, xrow, (uint32_t)orow};
    const Bind none = {NULL, 0};
    return n < VK_MV_TOKENS
         ? dispatch(v, v->pso_mv[x->kernel], c, 6, o, in, none, w, none, (uint32_t)((rows + v->mv_rows - 1) / v->mv_rows), (uint32_t)n, cached)
         : dispatch(v, v->pso[x->kernel], c, 6, o, in, none, w, none, (uint32_t)(rows + 63) / 64, (uint32_t)(n + 63) / 64,
                    cached);
}

/* out[t][r] (rows orow floats apart) = matrix x . the Q8_K rows of n tokens, one dispatch per part */
static int mm_dispatch(Vulkan *v, const VMat *x, Bind o, Bind in, int n, int orow, int cached) {
    int ok = 1;
    for (int p = 0; p < x->nparts && ok; p++) {
        const int r0 = p * x->part_rows, rows = x->rows - r0 < x->part_rows ? x->rows - r0 : x->part_rows;
        const Bind op = {o.b, o.off + (VkDeviceSize)r0 * 4}, w = {&x->part[p], 0};
        ok = mm_rows(v, x, w, rows, op, in, n, orow, (uint32_t)(x->K / QK_K) * Q8_BYTES, cached);
    }
    return ok;
}

int vk_has(const Vulkan *v, const uint8_t *w) {
    const VMat *x = v && !v->broken ? find_mat(v, w) : NULL;
    return x && x->nparts;
}

#define MAX_VK_JOBS 16

int vk_matmul(Vulkan *v, const VulkanJob *jobs, int n) {
    if (!v || v->broken || n < 1 || n > MAX_VK_JOBS) return 0;
    const VMat *mat[MAX_VK_JOBS];
    size_t xoff[MAX_VK_JOBS], ooff[MAX_VK_JOBS];
    int fresh[MAX_VK_JOBS];
    size_t xbytes = 0, obytes = 0;
    for (int i = 0; i < n; i++) {
        mat[i] = find_mat(v, jobs[i].w);
        if (!mat[i] || !mat[i]->nparts) return 0;
        /* jobs that read the same activations share one copy */
        fresh[i] = 1;
        for (int k = 0; k < i && fresh[i]; k++)
            if (jobs[k].xq == jobs[i].xq && jobs[k].nt == jobs[i].nt && mat[k]->K == mat[i]->K) {
                xoff[i] = xoff[k];
                fresh[i] = 0;
            }
        if (fresh[i]) {
            xoff[i] = xbytes;
            xbytes += ((size_t)jobs[i].nt * (size_t)(mat[i]->K / QK_K) * Q8_BYTES + 255) & ~(size_t)255;
        }
        ooff[i] = obytes;
        obytes += ((size_t)jobs[i].nt * (size_t)mat[i]->rows * 4 + 255) & ~(size_t)255;
    }
    if (!vbuf_reserve(v, &v->up, xbytes, HOST_UPLOAD) || !vbuf_reserve(v, &v->x, xbytes, HOST_NONE) ||
        !vbuf_reserve(v, &v->o, obytes, HOST_NONE) || !vbuf_reserve(v, &v->rb, obytes, HOST_READBACK))
        return 0;
    for (int i = 0; i < n; i++) {
        if (!fresh[i]) continue;
        const size_t nbk = (size_t)(mat[i]->K / QK_K);
        for (int t = 0; t < jobs[i].nt; t++) {
            const block_q8_K *src = jobs[i].xq + (size_t)t * (size_t)jobs[i].xstride;
            uint8_t *dst = v->up.map + xoff[i] + (size_t)t * nbk * Q8_BYTES;
            for (size_t b = 0; b < nbk; b++, dst += Q8_BYTES) {   /* to the GPU layout */
                memcpy(dst, src[b].qs, QK_K);
                memcpy(dst + Q8_GPU_D, &src[b].d, 4);
            }
        }
    }

    copy(v, &v->x, 0, &v->up, 0, xbytes);
    barrier(v, PS_TRANSFER, AC_TRANSFER_WRITE, PS_COMPUTE, AC_SHADER_READ | AC_SHADER_WRITE);
    int ok = 1;
    for (int i = 0; i < n && ok; i++) {
        const Bind o = {&v->o, ooff[i]}, xb = {&v->x, xoff[i]};
        ok = mm_dispatch(v, mat[i], o, xb, jobs[i].nt, mat[i]->rows, 0);
    }
    barrier(v, PS_COMPUTE, AC_SHADER_WRITE, PS_TRANSFER, AC_TRANSFER_READ);
    copy(v, &v->rb, 0, &v->o, 0, obytes);
    barrier(v, PS_TRANSFER, AC_TRANSFER_WRITE, PS_HOST, AC_HOST_READ);
    if (!submit(v) || !ok) {
        v->broken = 1;
        fprintf(stderr, "Vulkan: the GPU stopped responding, continuing on the CPU\n");
        return 0;
    }
    for (int i = 0; i < n; i++) {
        const float *src = (const float *)(v->rb.map + ooff[i]);
        const int rows = mat[i]->rows;
        for (int t = 0; t < jobs[i].nt; t++)
            memcpy(jobs[i].out + (size_t)t * (size_t)jobs[i].ostride, src + (size_t)t * rows, sizeof(float) * (size_t)rows);
    }
    return 1;
}

/* ------------------------------------------------------------------ whole forward pass
 *
 * As dx_dense_forward in gpu_d3d12.c, with the same constants: the forward pass of one token or
 * of a prompt chunk goes into one command buffer (kernels_dx_token.hlsl between the matmuls,
 * separated by compute barriers) and only the logits of the last token are read back. The GPU
 * keeps its own copy of the KV cache and of the convolution state, in the CPU layout. A
 * matmul's input and output are bound at offsets into the scratch buffer, which Vulkan wants
 * aligned: the input row of the logits, when it is not, is copied to an aligned place first. */

#define NONE_OFF 0xffffffffu

typedef struct {
    uint32_t attn_norm, ffn_norm, post_op, post_ffn, q_norm, k_norm, conv_w;   /* byte offsets in sw */
    const VMat *wq, *wk, *wv, *wo, *gate, *in_proj, *out_proj, *w_gate, *w_up, *w_down;
    /* Qwen: DeltaNet weights in sw, the layer's state in cs, the DeltaNet and shared-expert matrices */
    uint32_t ssm_conv, ssm_a, ssm_dt, ssm_norm, state;
    uint32_t router;                /* the router [n_expert][dim], then the shared expert's gate input, in sw */
    const VMat *ssm_qkv, *ssm_z, *ssm_alpha, *ssm_beta, *ssm_out, *sh_gate, *sh_up, *sh_down;
} VLayer;

typedef struct VDense {
    const Model *m;
    int n_ctx, B, D, hd, nh, nkv, kvd, F, lc, n_attn, n_conv;
    VLayer *layers;
    const VMat *output;
    uint32_t final_norm;
    VBuf scratch, kv, cs, sw;
    VBuf in, lrb;                   /* the tokens' input (host-visible, read by the shaders), logits */
    size_t kv_half;                 /* bytes of the key half of kv */
    size_t o_x, o_out, o_tmp, o_hq, o_big, o_gate, o_fg, o_fu, o_fq, o_sc, o_logits, o_last;
    size_t cs_bytes;                /* the convolution / DeltaNet state, as the CPU keeps it */
    /* Qwen (hybrid): the routed experts on the CPU, through moe() at every layer */
    int hybrid, qd, qkvw, vd, vh, kh, kconv, n_rot, Fs;
    int E;
    size_t o_h, o_z, o_al, o_be, o_q2, o_qg, o_kv, o_q, o_g, o_sh, o_rl, o_pl;
    size_t in_rope, in_y, in_s, hb_rl, hb_pl;   /* this pass's offsets in in and hrb */
    size_t xo_xg, xo_eg, xo_eu, xo_eq, xo_eo, o_ge;   /* the GPU's experts: inputs, gate, up, Q8_K, outputs, sum */
    size_t in_c;                    /* in: the table of their sum, [n][K] (row, weight) */
    int ge_on;                      /* o_ge holds the GPU's experts of the last callback, for the next add */
    VBuf hrb;                       /* h, the router logits and the prediction, read back at every layer */
    VkMoeFn moe;
    void *moe_arg;
    int cur_t0, cur_n, cur_split;   /* the rows of the running moe callback, and whether the pass is split */
    struct VExp *xp;                /* the experts in VRAM, or NULL */
} VDense;

static void xp_free(Vulkan *v);

static void dense_free(Vulkan *v) {
    VDense *t = v->dense;
    if (!t) return;
    VBuf *bufs[] = {&t->scratch, &t->kv, &t->cs, &t->sw, &t->in, &t->lrb, &t->hrb};
    for (size_t i = 0; i < sizeof(bufs) / sizeof(bufs[0]); i++) vbuf_free(v, bufs[i]);
    xp_free(v);
    free(t->layers);
    free(t);
    v->dense = NULL;
}

/* append n floats to the small-weights array, returning their byte offset (NONE_OFF for NULL) */
static uint32_t add_vec(float **arr, size_t *len, size_t *cap, const float *x, size_t n) {
    if (!x) return NONE_OFF;
    const size_t need = *len + ((n + 63) & ~(size_t)63);   /* keep offsets 256-byte aligned */
    if (need > *cap) {
        *cap = need * 2;
        *arr = (float *)realloc(*arr, sizeof(float) * *cap);
    }
    memcpy(*arr + *len, x, sizeof(float) * n);
    const uint32_t off = (uint32_t)(*len * 4);
    *len = need;
    return off;
}

static size_t carve(size_t *top, size_t bytes) {
    const size_t o = *top;
    *top += (bytes + 255) & ~(size_t)255;
    return o;
}

static int hybrid_init(Vulkan *v, const Model *m, int n_ctx, int n_batch, char *info, int infolen);

int vk_dense_init(Vulkan *v, const Model *m, int n_ctx, int n_batch, char *info, int infolen) {
    info[0] = 0;
    if (!v || v->broken) return 0;
    if (m->n_delta) return hybrid_init(v, m, n_ctx, n_batch, info, infolen);
    const int D = m->dim, hd = m->head_dim;
    int F = 0, ok = D % QK_K == 0 && hd % 2 == 0 && hd <= 512 && 256 % (hd / 2) == 0 && m->l_cache >= 2;
    const VMat *output = find_mat(v, m->output.data);
    if (output && !output->nparts) {
        snprintf(info, (size_t)infolen, "the output matrix is too large for the buffers of this GPU");
        return 0;
    }
    ok = ok && output;
    for (int l = 0; l < m->n_layer && ok; l++) {
        const Layer *L = &m->layers[l];
        if (L->is_moe) ok = 0;
        else if (L->w_gate.rows > F) F = L->w_gate.rows;
    }
    if (!ok || F % QK_K) {
        snprintf(info, (size_t)infolen, "this model needs the CPU for part of each token");
        return 0;
    }
    /* the matmul outputs at o_big + 4 D and + 4 (D + kvd) are bound there */
    const VkDeviceSize align = v->props.limits.minStorageBufferOffsetAlignment;
    if ((4 * (VkDeviceSize)D) % align || (4 * (VkDeviceSize)(D + m->n_kv_head * hd)) % align) {
        snprintf(info, (size_t)infolen, "the GPU needs %u-byte aligned buffer offsets", (unsigned)align);
        return 0;
    }
    VDense *t = (VDense *)calloc(1, sizeof(VDense));
    t->m = m;
    t->n_ctx = n_ctx;
    t->B = n_batch < 1 ? 1 : n_batch;
    t->D = D;
    t->hd = hd;
    t->nh = m->n_head;
    t->nkv = m->n_kv_head;
    t->kvd = m->n_kv_head * hd;
    t->F = F;
    t->lc = m->l_cache;
    t->n_attn = m->n_attn;
    t->n_conv = m->n_conv;
    t->output = output;
    t->layers = (VLayer *)calloc((size_t)m->n_layer, sizeof(VLayer));
    v->dense = t;

    float *sw = NULL;
    size_t len = 0, cap = 0;
    for (int l = 0; l < m->n_layer; l++) {
        const Layer *L = &m->layers[l];
        VLayer *X = &t->layers[l];
        X->attn_norm = add_vec(&sw, &len, &cap, L->attn_norm, (size_t)D);
        X->ffn_norm = add_vec(&sw, &len, &cap, L->ffn_norm, (size_t)D);
        X->post_op = add_vec(&sw, &len, &cap, L->post_op_norm, (size_t)D);
        X->post_ffn = add_vec(&sw, &len, &cap, L->post_ffn_norm, (size_t)D);
        if (L->is_attn) {
            X->q_norm = add_vec(&sw, &len, &cap, L->q_norm, (size_t)hd);
            X->k_norm = add_vec(&sw, &len, &cap, L->k_norm, (size_t)hd);
            X->wq = find_mat(v, L->wq.data);
            X->wk = find_mat(v, L->wk.data);
            X->wv = find_mat(v, L->wv.data);
            X->wo = find_mat(v, L->wo.data);
            X->gate = L->attn_gate.data ? find_mat(v, L->attn_gate.data) : NULL;
            ok = ok && X->wq && X->wk && X->wv && X->wo && (X->gate || !L->attn_gate.data) && L->q_norm && L->k_norm;
        } else {
            X->conv_w = add_vec(&sw, &len, &cap, L->conv_w, (size_t)D * (size_t)t->lc);
            X->in_proj = find_mat(v, L->in_proj.data);
            X->out_proj = find_mat(v, L->out_proj.data);
            ok = ok && X->in_proj && X->out_proj;
        }
        X->w_gate = find_mat(v, L->w_gate.data);
        X->w_up = find_mat(v, L->w_up.data);
        X->w_down = find_mat(v, L->w_down.data);
        ok = ok && X->w_gate && X->w_up && X->w_down && L->attn_norm && L->ffn_norm;
        const VMat *ms[] = {X->wq, X->wk, X->wv, X->wo, X->gate, X->in_proj, X->out_proj, X->w_gate, X->w_up, X->w_down};
        for (size_t i = 0; i < sizeof(ms) / sizeof(ms[0]) && ok; i++) ok = !ms[i] || ms[i]->nparts;
    }
    t->final_norm = add_vec(&sw, &len, &cap, m->final_norm, (size_t)D);
    if (!ok || !m->final_norm) {
        free(sw);
        dense_free(v);
        snprintf(info, (size_t)infolen, "some weights of this model are not on the GPU");
        return 0;
    }

    size_t top = 0;
    const size_t B = (size_t)t->B, big = 3 * D > D + 2 * t->kvd ? 3 * D : D + 2 * t->kvd;
    t->o_x = carve(&top, B * D * 4);
    t->o_out = carve(&top, B * D * 4);
    t->o_tmp = carve(&top, B * D * 4);
    t->o_hq = carve(&top, B * (D / QK_K) * Q8_BYTES);
    t->o_big = carve(&top, B * big * 4);
    t->o_gate = carve(&top, B * t->nh * 4);
    t->o_fg = carve(&top, B * F * 4);
    t->o_fu = carve(&top, B * F * 4);
    t->o_fq = carve(&top, B * (F / QK_K) * Q8_BYTES);
    t->o_sc = carve(&top, (size_t)t->nh * (size_t)n_ctx * 4);
    t->o_logits = carve(&top, (size_t)m->n_vocab * 4);
    t->o_last = carve(&top, (size_t)(D / QK_K) * Q8_BYTES);
    t->kv_half = (size_t)t->n_attn * (size_t)n_ctx * (size_t)t->kvd * 2;
    const size_t cs_bytes = (size_t)t->n_conv * (size_t)(t->lc - 1) * (size_t)D * 4;
    t->cs_bytes = cs_bytes;
    const size_t kv_bytes = t->kv_half ? 2 * t->kv_half : 256, max_buf = v->props.limits.maxStorageBufferRange;
    if (kv_bytes > max_buf || top > max_buf) {
        dense_free(v);
        free(sw);
        snprintf(info, (size_t)infolen, "a %d-token KV cache (%.0f MB) exceeds the %u MB buffer limit: lower -c", n_ctx,
                 kv_bytes / 1048576.0, (unsigned)(max_buf >> 20));
        return 0;
    }
    ok = vbuf_create(v, &t->scratch, top, HOST_NONE) && vbuf_create(v, &t->kv, kv_bytes, HOST_NONE) &&
         vbuf_create(v, &t->cs, cs_bytes ? cs_bytes : 256, HOST_NONE) && vbuf_create(v, &t->sw, len * 4, HOST_NONE) &&
         vbuf_create(v, &t->in, B * (size_t)(D + 2 * hd) * 4, HOST_UPLOAD) &&
         vbuf_create(v, &t->lrb, (size_t)(m->n_vocab + D) * 4, HOST_READBACK) && vbuf_reserve(v, &v->up, len * 4, HOST_UPLOAD);
    if (ok) {
        memcpy(v->up.map, sw, len * 4);
        copy(v, &t->sw, 0, &v->up, 0, len * 4);
        ok = submit(v);
    }
    free(sw);
    if (!ok) {
        dense_free(v);
        snprintf(info, (size_t)infolen, "not enough GPU memory for a %d-token context and %d-token batches", n_ctx, t->B);
        return 0;
    }
    snprintf(info, (size_t)infolen, "whole forward pass on the GPU, %.0f MB of KV cache, %.0f MB of activations",
             2.0 * t->kv_half / 1048576.0, top / 1048576.0);
    return 1;
}

#define XP_MAX_A 256        /* expert assignments per GPU launch */
#define XP_PAD 32           /* each buffer's rows of a launch start at a multiple of this */
#define XP_ROWS (XP_MAX_A + 8 * XP_PAD)   /* the rows of a launch, padding included */
static int xp_init(Vulkan *v, const Model *m, size_t used, double *mb);

/* Qwen3.5 / 3.6 (qwen35moe): every layer but its routed experts on the GPU. The state buffer is
 * the CPU's conv array (only DeltaNet state there), so vk_dense_conv copies it as it is. */
static int hybrid_init(Vulkan *v, const Model *m, int n_ctx, int n_batch, char *info, int infolen) {
    const int D = m->dim, hd = m->head_dim, d = m->ssm_d;
    const VMat *output = find_mat(v, m->output.data);
    int ok = D % QK_K == 0 && d == 128 && hd % 2 == 0 && hd <= 512 && m->n_rot <= hd && m->q_gate && m->ssm_conv_k >= 2 &&
             m->ssm_conv_k <= 8 && m->ff_shexp % QK_K == 0 && (m->n_head * hd) % QK_K == 0 && output && output->nparts &&
             v->props.limits.maxComputeWorkGroupInvocations >= 512 && v->props.limits.maxComputeWorkGroupSize[0] >= 512;   /* delta_rec */
    for (int l = 0; l < m->n_layer && ok; l++) {
        const Layer *L = &m->layers[l];
        ok = L->is_moe && L->sh_up.data && L->sh_inp && L->router.type == GGML_F32 &&
             (L->is_delta || L->is_attn) && !L->post_op_norm && !L->post_ffn_norm;
    }
    if (!ok) {
        snprintf(info, (size_t)infolen, "this model needs the CPU for part of each token");
        return 0;
    }
    VDense *t = (VDense *)calloc(1, sizeof(VDense));
    t->m = m;
    t->hybrid = 1;
    t->n_ctx = n_ctx;
    t->B = n_batch < 1 ? 1 : n_batch;
    t->D = D;
    t->hd = hd;
    t->nh = m->n_head;
    t->nkv = m->n_kv_head;
    t->kvd = m->n_kv_head * hd;
    t->qd = m->n_head * hd;
    t->kh = m->ssm_kh;
    t->vh = m->ssm_vh;
    t->vd = m->ssm_vh * d;
    t->qkvw = 2 * m->ssm_kh * d + t->vd;
    t->kconv = m->ssm_conv_k;
    t->n_rot = m->n_rot;
    t->Fs = m->ff_shexp;
    t->E = m->n_expert;
    t->n_attn = m->n_attn;
    t->output = output;
    t->layers = (VLayer *)calloc((size_t)m->n_layer, sizeof(VLayer));
    v->dense = t;
    const size_t layer_state = (size_t)(t->kconv - 1) * (size_t)t->qkvw + (size_t)t->vd * (size_t)d;   /* floats */
    t->cs_bytes = (size_t)m->n_delta * layer_state * 4;

    float *sw = NULL;
    size_t len = 0, cap = 0;
    for (int l = 0; l < m->n_layer && ok; l++) {
        const Layer *L = &m->layers[l];
        VLayer *X = &t->layers[l];
        X->attn_norm = add_vec(&sw, &len, &cap, L->attn_norm, (size_t)D);
        X->ffn_norm = add_vec(&sw, &len, &cap, L->ffn_norm, (size_t)D);
        if (L->is_attn) {
            X->q_norm = add_vec(&sw, &len, &cap, L->q_norm, (size_t)hd);
            X->k_norm = add_vec(&sw, &len, &cap, L->k_norm, (size_t)hd);
            X->wq = find_mat(v, L->wq.data);
            X->wk = find_mat(v, L->wk.data);
            X->wv = find_mat(v, L->wv.data);
            X->wo = find_mat(v, L->wo.data);
            ok = X->wq && X->wk && X->wv && X->wo && L->q_norm && L->k_norm && X->wq->rows == 2 * t->qd;
        } else {
            X->ssm_conv = add_vec(&sw, &len, &cap, L->ssm_conv, (size_t)t->qkvw * (size_t)t->kconv);
            X->ssm_a = add_vec(&sw, &len, &cap, L->ssm_a, (size_t)t->vh);
            X->ssm_dt = add_vec(&sw, &len, &cap, L->ssm_dt, (size_t)t->vh);
            X->ssm_norm = add_vec(&sw, &len, &cap, L->ssm_norm, (size_t)d);
            X->state = (uint32_t)((size_t)L->delta_idx * layer_state * 4);
            X->ssm_qkv = find_mat(v, L->ssm_qkv.data);
            X->ssm_z = find_mat(v, L->ssm_z.data);
            X->ssm_alpha = find_mat(v, L->ssm_alpha.data);
            X->ssm_beta = find_mat(v, L->ssm_beta.data);
            X->ssm_out = find_mat(v, L->ssm_out.data);
            ok = X->ssm_qkv && X->ssm_z && X->ssm_alpha && X->ssm_beta && X->ssm_out && L->ssm_conv && L->ssm_a && L->ssm_dt &&
                 L->ssm_norm;
        }
        {   /* the router's rows and the gate input as one f32 matrix */
            float *r = (float *)malloc(sizeof(float) * (size_t)(t->E + 1) * D);
            memcpy(r, L->router.data, sizeof(float) * (size_t)t->E * D);
            memcpy(r + (size_t)t->E * D, L->sh_inp, sizeof(float) * (size_t)D);
            X->router = add_vec(&sw, &len, &cap, r, (size_t)(t->E + 1) * D);
            free(r);
        }
        X->sh_gate = find_mat(v, L->sh_gate.data);
        X->sh_up = find_mat(v, L->sh_up.data);
        X->sh_down = find_mat(v, L->sh_down.data);
        ok = ok && X->sh_gate && X->sh_up && X->sh_down && L->attn_norm && L->ffn_norm;
        const VMat *ms[] = {X->wq, X->wk, X->wv, X->wo, X->ssm_qkv, X->ssm_z, X->ssm_alpha, X->ssm_beta, X->ssm_out,
                            X->sh_gate, X->sh_up, X->sh_down};
        for (size_t i = 0; i < sizeof(ms) / sizeof(ms[0]) && ok; i++) ok = !ms[i] || ms[i]->nparts;
    }
    t->final_norm = add_vec(&sw, &len, &cap, m->final_norm, (size_t)D);
    if (!ok || !m->final_norm) {
        free(sw);
        dense_free(v);
        snprintf(info, (size_t)infolen, "some weights of this model are not on the GPU");
        return 0;
    }

    size_t top = 0;
    const size_t B = (size_t)t->B, qd = (size_t)t->qd, wide = qd > (size_t)t->vd ? qd : (size_t)t->vd;
    t->o_x = carve(&top, B * D * 4);
    t->o_out = carve(&top, B * D * 4);
    t->o_h = carve(&top, B * D * 4);
    t->o_sh = carve(&top, B * D * 4);
    t->o_hq = carve(&top, B * (D / QK_K) * Q8_BYTES);
    t->o_big = carve(&top, B * (size_t)t->qkvw * 4);
    t->o_z = carve(&top, B * (size_t)t->vd * 4);
    t->o_al = carve(&top, B * (size_t)t->vh * 4);
    t->o_be = carve(&top, B * (size_t)t->vh * 4);
    t->o_tmp = carve(&top, B * wide * 4);
    t->o_q2 = carve(&top, B * (wide / QK_K) * Q8_BYTES);
    t->o_qg = carve(&top, B * 2 * qd * 4);
    t->o_kv = carve(&top, B * 2 * (size_t)t->kvd * 4);
    t->o_q = carve(&top, B * qd * 4);
    t->o_g = carve(&top, B * qd * 4);
    t->o_fg = carve(&top, B * (size_t)t->Fs * 4);
    t->o_fu = carve(&top, B * (size_t)t->Fs * 4);
    t->o_fq = carve(&top, B * (size_t)(t->Fs / QK_K) * Q8_BYTES);
    t->o_rl = carve(&top, B * (size_t)(t->E + 1) * 4);
    {   /* the GPU's experts: XP_MAX_A assignments */
        const size_t A = XP_ROWS, F = (size_t)m->ff_exp;
        t->xo_xg = carve(&top, A * (D / QK_K) * Q8_BYTES);
        t->xo_eg = carve(&top, A * F * 4);
        t->xo_eu = carve(&top, A * F * 4);
        t->xo_eq = carve(&top, A * (F / QK_K) * Q8_BYTES);
        t->xo_eo = carve(&top, A * (size_t)D * 4);
        t->o_ge = carve(&top, B * (size_t)D * 4);
    }
    t->o_pl = carve(&top, (size_t)t->E * 4);
    t->o_logits = carve(&top, (size_t)m->n_vocab * 4);
    t->o_last = carve(&top, (size_t)(D / QK_K) * Q8_BYTES);
    t->kv_half = (size_t)t->n_attn * (size_t)n_ctx * (size_t)t->kvd * 2;
    const size_t kv_bytes = t->kv_half ? 2 * t->kv_half : 256, max_buf = v->props.limits.maxStorageBufferRange;
    const VkDeviceSize align = v->props.limits.minStorageBufferOffsetAlignment;
    if (kv_bytes > max_buf || top > max_buf || t->cs_bytes > max_buf || len * 4 > max_buf || (4 * (VkDeviceSize)t->kvd) % align) {
        dense_free(v);
        free(sw);
        snprintf(info, (size_t)infolen, "a %d-token KV cache (%.0f MB) exceeds the %u MB buffer limit: lower -c", n_ctx,
                 kv_bytes / 1048576.0, (unsigned)(max_buf >> 20));
        return 0;
    }
    ok = vbuf_create(v, &t->scratch, top, HOST_NONE) && vbuf_create(v, &t->kv, kv_bytes, HOST_NONE) &&
         vbuf_create(v, &t->cs, t->cs_bytes, HOST_NONE) && vbuf_create(v, &t->sw, len * 4, HOST_NONE) &&
         vbuf_create(v, &t->in, B * (size_t)(2 * D + hd + 1) * 4 + B * (size_t)m->n_expert_used * 8 + 256, HOST_UPLOAD) &&
         vbuf_create(v, &t->lrb, (size_t)(m->n_vocab + D) * 4, HOST_READBACK) &&
         vbuf_create(v, &t->hrb, (B * (D + t->E + 1) + t->E) * 4, HOST_READBACK) && vbuf_reserve(v, &v->up, len * 4, HOST_UPLOAD);
    if (ok) {
        memcpy(v->up.map, sw, len * 4);
        copy(v, &t->sw, 0, &v->up, 0, len * 4);
        ok = submit(v);
    }
    free(sw);
    if (!ok) {
        dense_free(v);
        snprintf(info, (size_t)infolen, "not enough GPU memory for a %d-token context and %d-token batches", n_ctx, t->B);
        return 0;
    }
    const size_t used = top + kv_bytes + t->cs_bytes + len * 4;
    double xmb = 0;
    const int nx = xp_init(v, m, used, &xmb);
    char xs[96] = "";
    if (nx) snprintf(xs, sizeof(xs), "; up to %d of the most used ones in VRAM (%.0f MB)", nx, xmb);
    snprintf(info, (size_t)infolen, "every layer on the GPU, the routed experts on the CPU%s, %.0f MB of KV cache, "
             "%.0f MB of state, %.0f MB of activations", xs, 2.0 * t->kv_half / 1048576.0, t->cs_bytes / 1048576.0,
             top / 1048576.0);
    return 1;
}

static void compute_barrier(Vulkan *v) {
    barrier(v, PS_COMPUTE, AC_SHADER_WRITE, PS_COMPUTE, AC_SHADER_READ | AC_SHADER_WRITE);
}

/* out[t][r] (scratch offset oo, rows orow floats apart) = matrix x . the Q8_K rows of n tokens at xo */
static int tok_mm(Vulkan *v, const VMat *x, size_t xo, size_t oo, int n, int orow) {
    const Bind o = {&v->dense->scratch, oo}, in = {&v->dense->scratch, xo};
    return mm_dispatch(v, x, o, in, n, orow, 1);
}

static int tok_run(Vulkan *v, int k, const uint32_t c[16], uint32_t groups, uint32_t tokens) {
    VDense *t = v->dense;
    const Bind sc = {&t->scratch, 0}, kv = {&t->kv, 0}, cs = {&t->cs, 0}, sw = {&t->sw, 0}, in = {&t->in, 0};
    const int ok = dispatch(v, v->pso_tok[k], c, 16, sc, kv, cs, sw, in, groups, tokens, 1);
    compute_barrier(v);
    return ok;
}

static uint32_t fbits(float f) {
    uint32_t u;
    memcpy(&u, &f, 4);
    return u;
}

/* x = norm (mode 0: embedding from the input; 1: x += out, RMS-normalized by post) -> Q8_K at hq */
static int tok_resid(Vulkan *v, uint32_t mode, uint32_t post, uint32_t norm, int n) {
    const VDense *t = v->dense;
    const uint32_t c[16] = {(uint32_t)t->D, (uint32_t)t->o_x, (uint32_t)t->o_out, (uint32_t)t->o_hq, mode, post, norm,
                            fbits(t->m->eps)};
    return tok_run(v, T_RESID, c, 1, (uint32_t)n);
}

/* the end of a pass: the logits of the last token (the Q8_K row of its final norm at o_hq), then
 * the submit and the read back */
static int finish_pass(Vulkan *v, int ok, int n, float *logits, float *x_last) {
    VDense *t = v->dense;
    const Model *m = t->m;
    const int D = t->D;
    const uint32_t nb = (uint32_t)(D / QK_K);
    if (ok && logits) {
        size_t xo = t->o_hq + (size_t)(n - 1) * nb * Q8_BYTES;
        if (xo % v->props.limits.minStorageBufferOffsetAlignment) {   /* not bindable there: move the row */
            barrier(v, PS_COMPUTE, AC_SHADER_WRITE, PS_TRANSFER, AC_TRANSFER_READ);
            copy(v, &t->scratch, t->o_last, &t->scratch, xo, (VkDeviceSize)nb * Q8_BYTES);
            barrier(v, PS_TRANSFER, AC_TRANSFER_WRITE, PS_COMPUTE, AC_SHADER_READ);
            xo = t->o_last;
        }
        ok = tok_mm(v, t->output, xo, t->o_logits, 1, m->n_vocab);
        barrier(v, PS_COMPUTE, AC_SHADER_WRITE, PS_TRANSFER, AC_TRANSFER_READ);
        copy(v, &t->lrb, 0, &t->scratch, t->o_logits, (VkDeviceSize)m->n_vocab * 4);
        if (x_last) copy(v, &t->lrb, (VkDeviceSize)m->n_vocab * 4, &t->scratch, t->o_x + (size_t)(n - 1) * D * 4, (VkDeviceSize)D * 4);
        barrier(v, PS_TRANSFER, AC_TRANSFER_WRITE, PS_HOST, AC_HOST_READ);
    }
    if (!submit(v) || !ok) {
        v->broken = 1;
        fprintf(stderr, "Vulkan: the GPU stopped responding\n");
        return 0;
    }
    if (logits) memcpy(logits, t->lrb.map, sizeof(float) * (size_t)m->n_vocab);
    if (logits && x_last) memcpy(x_last, t->lrb.map + 4 * (size_t)m->n_vocab, sizeof(float) * (size_t)D);
    return 1;
}

/* qwen35moe. The buffers of a pass hold its tokens' rows; the functions below work on the rows of
 * the tokens [t0, t0 + n). */

/* x += s * shared + y (mode 2), or x += out (mode 1), or the embeddings (mode 0, t0 = 0); the Q8_K
 * of its norm at o_hq, and the floats at h unless NONE_OFF */
static int moe_resid(Vulkan *v, uint32_t mode, uint32_t norm, size_t out, uint32_t h, int t0, int n) {
    VDense *t = v->dense;
    const size_t r = (size_t)t0 * t->D * 4;
    const uint32_t c[16] = {(uint32_t)t->D, (uint32_t)(t->o_x + r), (uint32_t)(out + r),
                            (uint32_t)(t->o_hq + (size_t)t0 * (t->D / QK_K) * Q8_BYTES), mode, norm, fbits(t->m->eps),
                            h == NONE_OFF ? NONE_OFF : (uint32_t)(h + r), (uint32_t)(t->in_y + r), (uint32_t)(t->in_s + 4 * (size_t)t0),
                            mode == 2 && t->ge_on ? (uint32_t)(t->o_ge + r) : NONE_OFF};
    if (mode == 2) t->ge_on = 0;   /* added */
    return tok_run(v, T_MOE_RESID, c, 1, (uint32_t)n);
}

/* Q8_K rows of the floats at o_tmp (width floats per token) into o_q2 */
static int quant_tmp(Vulkan *v, int width, int t0, int n) {
    const VDense *t = v->dense;
    const uint32_t nbw = (uint32_t)(width / QK_K);
    const uint32_t c[16] = {(uint32_t)(t->o_tmp + (size_t)t0 * width * 4), (uint32_t)(t->o_q2 + (size_t)t0 * nbw * Q8_BYTES),
                            (uint32_t)width * 4, nbw * Q8_BYTES};
    return tok_run(v, T_QUANT, c, nbw, (uint32_t)n);
}

/* layer l up to the routed experts: the residual add of layer l - 1's (from the CPU, in IN), the
 * mixer, its residual add and the router (for one token also the next layer's, to read its
 * experts ahead); then h and the logits are copied to hrb */
static int hybrid_mixer(Vulkan *v, int l, int t0, int n, int pos0) {
    VDense *t = v->dense;
    const Model *m = t->m;
    const Layer *L = &m->layers[l];
    const VLayer *X = &t->layers[l];
    const int D = t->D, hd = t->hd, qd = t->qd, kvd = t->kvd, vd = t->vd, vh = t->vh;
    const uint32_t un = (uint32_t)n, E1 = (uint32_t)t->E + 1;
    const size_t T = (size_t)t0, hq = t->o_hq + T * (D / QK_K) * Q8_BYTES, out = t->o_out + T * D * 4;
    int ok = l == 0 || moe_resid(v, 2, X->attn_norm, t->o_sh, NONE_OFF, t0, n);
    if (L->is_delta) {
        const size_t big = t->o_big + T * t->qkvw * 4, z = t->o_z + T * vd * 4, tmp = t->o_tmp + T * vd * 4;
        const size_t al = t->o_al + T * vh * 4, be = t->o_be + T * vh * 4;
        ok = ok && tok_mm(v, X->ssm_qkv, hq, big, n, t->qkvw) && tok_mm(v, X->ssm_z, hq, z, n, vd) &&
             tok_mm(v, X->ssm_alpha, hq, al, n, vh) && tok_mm(v, X->ssm_beta, hq, be, n, vh);
        compute_barrier(v);
        const uint32_t cc[16] = {(uint32_t)t->qkvw, (uint32_t)big, (uint32_t)t->qkvw * 4, un, X->ssm_conv, X->state,
                                 (uint32_t)t->kconv};
        const uint32_t cr[16] = {(uint32_t)big, (uint32_t)t->qkvw * 4, (uint32_t)z, (uint32_t)vd * 4, (uint32_t)al, (uint32_t)be,
                                 (uint32_t)tmp, un, X->state + (uint32_t)(4 * (t->kconv - 1) * t->qkvw), (uint32_t)t->kh,
                                 (uint32_t)vh, fbits(m->eps), X->ssm_a, X->ssm_dt, X->ssm_norm, fbits(1.f / sqrtf((float)m->ssm_d))};
        ok = ok && tok_run(v, T_DCONV, cc, (uint32_t)(t->qkvw + 255) / 256, 1) && tok_run(v, T_DREC, cr, (uint32_t)vh, 1) &&
             quant_tmp(v, vd, t0, n) && tok_mm(v, X->ssm_out, t->o_q2 + T * (vd / QK_K) * Q8_BYTES, out, n, D);
    } else {
        const size_t qg = t->o_qg + T * 2 * qd * 4, kv = t->o_kv + T * 2 * kvd * 4, q = t->o_q + T * qd * 4;
        const size_t g = t->o_g + T * qd * 4, tmp = t->o_tmp + T * qd * 4;
        ok = ok && tok_mm(v, X->wq, hq, qg, n, 2 * qd) && tok_mm(v, X->wk, hq, kv, n, 2 * kvd) &&
             tok_mm(v, X->wv, hq, kv + 4 * (size_t)kvd, n, 2 * kvd);
        compute_barrier(v);
        const size_t kbase = (size_t)L->attn_idx * (size_t)t->n_ctx * (size_t)kvd * 2;
        const size_t krow = kbase + (size_t)pos0 * (size_t)kvd * 2;
        const uint32_t cp[16] = {(uint32_t)hd, (uint32_t)qg, (uint32_t)kv, (uint32_t)t->nh, X->q_norm, X->k_norm,
                                 (uint32_t)(t->in_rope + T * hd * 4), fbits(m->eps), (uint32_t)krow,
                                 (uint32_t)(t->kv_half + krow), (uint32_t)t->n_rot, (uint32_t)q, (uint32_t)g, (uint32_t)qd,
                                 (uint32_t)kvd};
        const uint32_t ca[16] = {(uint32_t)hd, (uint32_t)q, (uint32_t)g, (uint32_t)tmp, (uint32_t)kbase,
                                 (uint32_t)(t->kv_half + kbase), (uint32_t)kvd, (uint32_t)qd, (uint32_t)pos0,
                                 (uint32_t)(t->nh / t->nkv), 0, 0, fbits(1.f / sqrtf((float)hd))};
        ok = ok && tok_run(v, T_QPREP, cp, (uint32_t)(t->nh + t->nkv), un) && tok_run(v, T_QATTN, ca, (uint32_t)t->nh, un) &&
             quant_tmp(v, qd, t0, n) && tok_mm(v, X->wo, t->o_q2 + T * (qd / QK_K) * Q8_BYTES, out, n, D);
    }
    compute_barrier(v);
    ok = ok && moe_resid(v, 1, X->ffn_norm, t->o_out, (uint32_t)t->o_h, t0, n);
    const int pred = n == 1 && l + 1 < m->n_layer;
    const size_t h = t->o_h + T * D * 4, rl = t->o_rl + T * E1 * 4;
    const uint32_t cl[16] = {(uint32_t)D, (uint32_t)h, un, (uint32_t)rl, X->router, E1, E1, NONE_OFF};
    ok = ok && tok_run(v, T_ROUTER, cl, (E1 + 7) / 8, (un + 7) / 8);
    if (pred) {
        const uint32_t cn[16] = {(uint32_t)D, (uint32_t)(t->o_x + T * D * 4), 1, (uint32_t)t->o_pl, t->layers[l + 1].router,
                                 (uint32_t)t->E, (uint32_t)t->E, t->layers[l + 1].ffn_norm, fbits(m->eps)};
        ok = ok && tok_run(v, T_ROUTER, cn, (uint32_t)(t->E + 7) / 8, 1);
    }
    barrier(v, PS_COMPUTE, AC_SHADER_WRITE, PS_TRANSFER, AC_TRANSFER_READ);
    copy(v, &t->hrb, T * D * 4, &t->scratch, h, (VkDeviceSize)n * D * 4);
    copy(v, &t->hrb, t->hb_rl + T * E1 * 4, &t->scratch, rl, (VkDeviceSize)n * E1 * 4);
    if (pred) copy(v, &t->hrb, t->hb_pl, &t->scratch, t->o_pl, (VkDeviceSize)t->E * 4);
    barrier(v, PS_TRANSFER, AC_TRANSFER_WRITE, PS_HOST, AC_HOST_READ);
    return ok;
}

/* the shared expert of layer l into o_sh (scaled on the next residual add by the CPU's gate) */
static int hybrid_shared(Vulkan *v, int l, int t0, int n) {
    VDense *t = v->dense;
    const VLayer *X = &t->layers[l];
    const size_t T = (size_t)t0, Fs = (size_t)t->Fs;
    const size_t hq = t->o_hq + T * (t->D / QK_K) * Q8_BYTES, fg = t->o_fg + T * Fs * 4, fu = t->o_fu + T * Fs * 4;
    const size_t fq = t->o_fq + T * (Fs / QK_K) * Q8_BYTES;
    const uint32_t cs[16] = {(uint32_t)fg, (uint32_t)fu, (uint32_t)fq, (uint32_t)Fs};
    int ok = tok_mm(v, X->sh_gate, hq, fg, n, t->Fs) && tok_mm(v, X->sh_up, hq, fu, n, t->Fs);
    compute_barrier(v);
    ok = ok && tok_run(v, T_SILU, cs, (uint32_t)(Fs / QK_K), (uint32_t)n) &&
         tok_mm(v, X->sh_down, fq, t->o_sh + T * t->D * 4, n, t->D);
    compute_barrier(v);
    return ok;
}

/* the CPU's part of layer l: the routed experts, from what hybrid_mixer read back */
static void hybrid_moe(Vulkan *v, int l, int t0, int n) {
    VDense *t = v->dense;
    const size_t D = (size_t)t->D, E1 = (size_t)t->E + 1;
    const float *hb = (const float *)t->hrb.map;
    const int pred = n == 1 && l + 1 < t->m->n_layer;
    t->cur_t0 = t0;
    t->cur_n = n;
    t->moe(t->moe_arg, l, hb + (size_t)t0 * D, hb + t->hb_rl / 4 + (size_t)t0 * E1, pred ? hb + t->hb_pl / 4 : NULL, n,
           (float *)(t->in.map + t->in_y) + (size_t)t0 * D, (float *)(t->in.map + t->in_s) + t0);
}

/* qwen35moe: the GPU stops at every layer for the CPU's experts. A pass of 32 tokens or more is
 * cut in two halves (PITHOS_VK_NOSPLIT: not), and the GPU computes a layer of one half while the
 * CPU computes the experts of the other. */
static int hybrid_forward(Vulkan *v, const float *x, int n, const float *rope_cs, int pos0, float *logits, float *x_last) {
    VDense *t = v->dense;
    const Model *m = t->m;
    const int D = t->D, hd = t->hd, nl = m->n_layer;
    t->in_rope = 4 * (size_t)n * D;
    t->in_y = t->in_rope + 4 * (size_t)n * hd;
    t->in_s = t->in_y + 4 * (size_t)n * D;
    t->in_c = (t->in_s + 4 * (size_t)n + 255) & ~(size_t)255;
    t->ge_on = 0;
    t->hb_rl = 4 * (size_t)n * D;
    t->hb_pl = t->hb_rl + 4 * (size_t)n * (t->E + 1);
    memcpy(t->in.map, x, sizeof(float) * (size_t)n * D);
    memcpy(t->in.map + t->in_rope, rope_cs, sizeof(float) * (size_t)n * hd);
    int ok = moe_resid(v, 0, t->layers[0].attn_norm, t->o_out, NONE_OFF, 0, n);
    /* the first half: a multiple of 16 tokens, so that its rows of every buffer stay bindable */
    const int na = n >= 32 && !getenv("PITHOS_VK_NOSPLIT") ? (n / 2 + 15) & ~15 : n, nb = n - na;
    t->cur_split = nb > 0;   /* the GPU is busy with the other half: the CPU computes every expert */
    if (!nb) {
        for (int l = 0; l < nl && ok; l++) {
            ok = hybrid_mixer(v, l, 0, n, pos0) && hybrid_shared(v, l, 0, n) && submit(v);
            if (ok) hybrid_moe(v, l, 0, n);
            ok = ok && !v->broken;   /* the experts on the GPU failed */
        }
    } else {
        ok = ok && hybrid_mixer(v, 0, 0, na, pos0) && hybrid_shared(v, 0, 0, na) && submit(v);
        for (int l = 0; l < nl && ok; l++) {
            ok = hybrid_mixer(v, l, na, nb, pos0 + na) && hybrid_shared(v, l, na, nb) && kick(v);
            if (ok) hybrid_moe(v, l, 0, na);
            ok = wait_gpu(v) && ok;
            if (l + 1 < nl) ok = ok && hybrid_mixer(v, l + 1, 0, na, pos0) && hybrid_shared(v, l + 1, 0, na) && kick(v);
            if (ok) hybrid_moe(v, l, na, nb);
            ok = wait_gpu(v) && ok;
        }
    }
    ok = ok && moe_resid(v, 2, t->final_norm, t->o_sh, NONE_OFF, 0, n);
    ok = finish_pass(v, ok, n, logits, x_last);
    v->ln = &v->lanes[1];
    if (!wait_gpu(v)) ok = 0;   /* it ran before the pass's end: only its fence is left */
    v->ln = &v->lanes[0];
    return ok;
}

/* ------------------------------------------------------------------ routed experts in VRAM */

/* Slots of whole experts (gate, up, down in the GPU layout) in buffers of up to 1 GB. The
 * policy is gpu_opencl.c's in short: every routed use adds 1 to the expert's score, every
 * vk_rebalance multiplies the scores by XP_DECAY and moves the best scored experts in, into a
 * free slot or over the lowest scored resident one. Its experts were just used, so their pages
 * are in RAM: packing them costs no disk read. The copies run at the start of the next pass.
 * A launch runs on lane 1 while the CPU computes the other experts: three matrix-vector
 * dispatches per buffer (gate, up, down; MV_EXPERTS reads each assignment's matrix and input
 * row from a table), one SwiGLU over all of them, then their weighted sum per token into o_ge,
 * which the next residual add (moe_resid) takes; the CPU never waits for it. */
#define XP_STAGE 16         /* uploads per vk_rebalance at most */
#define XP_DECAY 0.99f

typedef struct VExp {
    int nl, E, n_slots, n_free, per_buf;
    size_t slot_bytes;
    VBuf *bufs;
    VMat *tm;                       /* [layer][gate, up, down]: kernels and row sizes */
    size_t *toff;                   /* [layer][3]: byte offsets in a slot */
    int *slot_of, *owner;           /* [layer * E + e] -> slot or -1; [slot] -> layer * E + e or -1 */
    float *score;                   /* [layer * E + e] */
    VBuf stage, tab;                /* packed experts on their way up; the launch's tables */
} VExp;

static void xp_free(Vulkan *v) {
    VExp *p = v->dense ? v->dense->xp : NULL;
    if (!p) return;
    for (int i = 0; p->per_buf && i < (p->n_slots + p->per_buf - 1) / p->per_buf; i++) vbuf_free(v, &p->bufs[i]);
    vbuf_free(v, &p->stage);
    vbuf_free(v, &p->tab);
    free(p->bufs);
    free(p->tm);
    free(p->toff);
    free(p->slot_of);
    free(p->owner);
    free(p->score);
    free(p);
    v->dense->xp = NULL;
}

/* the slots that fit in the GPU memory left (used: bytes the pass already holds besides the
 * weights) less a reserve, or PITHOS_VK_EXPERT_MB (0: none); discrete GPUs only */
static int xp_init(Vulkan *v, const Model *m, size_t used, double *mb) {
    VDense *t = v->dense;
    const char *env = getenv("PITHOS_VK_EXPERT_MB");
    if (v->props.deviceType != 2 && !env) return 0;
    VExp *p = (VExp *)calloc(1, sizeof(VExp));
    t->xp = p;
    p->nl = m->n_layer;
    p->E = m->n_expert;
    p->tm = (VMat *)calloc((size_t)p->nl * 3, sizeof(VMat));
    p->toff = (size_t *)calloc((size_t)p->nl * 3, sizeof(size_t));
    int ok = 1;
    for (int l = 0; l < p->nl && ok; l++) {
        const Layer *L = &m->layers[l];
        const Mat *ws[3] = {&L->gate_exps, &L->up_exps, &L->down_exps};
        size_t o = 0;
        for (int k = 0; k < 3 && ok; k++) {
            ok = vmat_init(&p->tm[l * 3 + k], ws[k], ws[k]->data);
            p->toff[l * 3 + k] = o;
            o += (p->tm[l * 3 + k].gpu_row * (size_t)ws[k]->rows + 255) & ~(size_t)255;
        }
        if (o > p->slot_bytes) p->slot_bytes = o;
    }
    size_t weights = 0;
    for (int i = 0; i < v->nmats; i++)
        if (v->mats[i].nparts) weights += v->mats[i].gpu_row * (size_t)v->mats[i].rows;
    const double heap = (double)device_memory(&v->vk, v->pd), have = heap - (double)weights - (double)used - 512.0 * 1048576;
    const double budget = env ? atof(env) * 1048576.0 : have;
    int want = ok && budget > 0 ? (int)(budget / (double)p->slot_bytes) : 0;
    if (want > p->nl * p->E) want = p->nl * p->E;
    const VkDeviceSize lim = v->props.limits.maxStorageBufferRange < (1u << 30) ? v->props.limits.maxStorageBufferRange : (1u << 30);
    p->per_buf = (int)(lim / p->slot_bytes);
    if (p->per_buf < 1 || (want + p->per_buf - 1) / p->per_buf > 8) want = 0;   /* XP_ROWS has room for 8 buffers */
    p->bufs = (VBuf *)calloc(8, sizeof(VBuf));
    for (int b = 0; b * p->per_buf < want; b++) {   /* as many buffers as the driver gives */
        const int k = want - b * p->per_buf < p->per_buf ? want - b * p->per_buf : p->per_buf;
        if (!vbuf_create(v, &p->bufs[b], (VkDeviceSize)k * p->slot_bytes, HOST_NONE)) break;
        p->n_slots += k;
    }
    ok = p->n_slots > 0 && vbuf_create(v, &p->stage, (VkDeviceSize)XP_STAGE * p->slot_bytes, HOST_UPLOAD) &&
         vbuf_create(v, &p->tab, (VkDeviceSize)3 * XP_ROWS * 8, HOST_UPLOAD);
    if (!ok) {
        xp_free(v);
        return 0;
    }
    const size_t ne = (size_t)p->nl * p->E;
    p->slot_of = (int *)malloc(sizeof(int) * ne);
    p->score = (float *)calloc(ne, sizeof(float));
    p->owner = (int *)malloc(sizeof(int) * (size_t)p->n_slots);
    for (size_t i = 0; i < ne; i++) p->slot_of[i] = -1;
    for (int i = 0; i < p->n_slots; i++) p->owner[i] = -1;
    p->n_free = p->n_slots;
    *mb = (double)p->n_slots * p->slot_bytes / 1048576.0;
    return p->n_slots;
}

static VExp *xp_of(const Vulkan *v) {
    return v && v->dense && !v->broken ? v->dense->xp : NULL;
}

int vk_experts_on(const Vulkan *v) {
    return xp_of(v) != NULL;
}

int vk_expert_resident(const Vulkan *v, int layer, int expert) {
    const VExp *p = xp_of(v);
    return p && layer < p->nl && p->slot_of[(size_t)layer * p->E + expert] >= 0;
}

int vk_moe_ready(const Vulkan *v, int n) {
    return xp_of(v) && !v->dense->cur_split && n <= XP_MAX_A && !v->lanes[0].recording && !v->lanes[0].inflight;
}

void vk_note_usage(Vulkan *v, int layer, int expert) {
    VExp *p = xp_of(v);
    if (p && layer < p->nl) p->score[(size_t)layer * p->E + expert] += 1.f;
}

/* matrix x of the experts in buffer w for n assignments: their matrices and input rows in the
 * table at tab (pairs of uints), outputs orow floats apart at o */
static int mvx(Vulkan *v, const VMat *x, Bind w, Bind tab, int rows, Bind o, Bind in, int n, int orow, uint32_t xrow) {
    const uint32_t c[6] = {(uint32_t)rows, (uint32_t)(x->K / QK_K), (uint32_t)n, (uint32_t)x->gpu_row, xrow, (uint32_t)orow};
    const Bind none = {NULL, 0};
    return dispatch(v, v->pso_mvx[x->kernel], c, 6, o, in, none, w, tab, (uint32_t)(rows + 7) / 8, (uint32_t)n, 0);
}

int vk_moe_begin(Vulkan *v, int layer, const int *a_expert, const int *a_tok, const float *a_w, int n) {
    VExp *p = xp_of(v);
    if (!p || n < 1 || n > XP_MAX_A) return 0;
    VDense *t = v->dense;
    v->ln = &v->lanes[1];   /* lane 0 stays free for the next layer, recorded while this runs */
    if (!wait_gpu(v)) {
        v->ln = &v->lanes[0];
        v->broken = 1;
        return 0;
    }
    const int D = t->D, F = p->tm[layer * 3].rows, nbufs = (p->n_slots + p->per_buf - 1) / p->per_buf;
    const size_t nbq = (size_t)(D / QK_K) * Q8_BYTES, nbf = (size_t)(F / QK_K) * Q8_BYTES;
    /* rows by buffer: those of buffer b from rb[b], a multiple of XP_PAD */
    int cnt[8] = {0}, rb[9], fill[8] = {0};
    for (int i = 0; i < n; i++) cnt[p->slot_of[(size_t)layer * p->E + a_expert[i]] / p->per_buf]++;
    rb[0] = 0;
    for (int b = 0; b < nbufs; b++) rb[b + 1] = (rb[b] + cnt[b] + XP_PAD - 1) / XP_PAD * XP_PAD;
    uint32_t *tab = (uint32_t *)p->tab.map;   /* [gate, up, down][XP_ROWS][matrix offset, input row] */
    const int K = t->m->n_expert_used;
    uint32_t *ct = (uint32_t *)(t->in.map + t->in_c);   /* the sum: [token][K] (row, weight) */
    int nk[XP_MAX_A] = {0};
    for (int i = 0; i < t->cur_n * K; i++) ct[2 * i] = NONE_OFF;
    for (int i = 0; i < n; i++) {
        const int slot = p->slot_of[(size_t)layer * p->E + a_expert[i]], b = slot / p->per_buf, r = rb[b] + fill[b]++;
        const uint32_t base = (uint32_t)((size_t)(slot % p->per_buf) * p->slot_bytes);
        const int tk = a_tok[i];
        ct[2 * (tk * K + nk[tk])] = (uint32_t)r;
        memcpy(&ct[2 * (tk * K + nk[tk]++) + 1], &a_w[i], 4);
        for (int k = 0; k < 3; k++) {
            tab[2 * (k * XP_ROWS + r)] = base + (uint32_t)p->toff[layer * 3 + k];
            tab[2 * (k * XP_ROWS + r) + 1] = k < 2 ? (uint32_t)(t->cur_t0 + a_tok[i]) : (uint32_t)r;
        }
    }
    int ok = 1;
    for (int b = 0; b < nbufs && ok; b++) {
        if (!cnt[b]) continue;
        const Bind w = {&p->bufs[b], 0}, in = {&t->scratch, t->o_hq};
        const Bind tg = {&p->tab, (VkDeviceSize)rb[b] * 8}, tu = {&p->tab, (VkDeviceSize)(XP_ROWS + rb[b]) * 8};
        const Bind og = {&t->scratch, t->xo_eg + (size_t)rb[b] * F * 4}, ou = {&t->scratch, t->xo_eu + (size_t)rb[b] * F * 4};
        ok = mvx(v, &p->tm[layer * 3], w, tg, F, og, in, cnt[b], F, (uint32_t)nbq) &&
             mvx(v, &p->tm[layer * 3 + 1], w, tu, F, ou, in, cnt[b], F, (uint32_t)nbq);
    }
    compute_barrier(v);
    const uint32_t cs[16] = {(uint32_t)t->xo_eg, (uint32_t)t->xo_eu, (uint32_t)t->xo_eq, (uint32_t)F};
    ok = ok && tok_run(v, T_SILU, cs, (uint32_t)(F / QK_K), (uint32_t)rb[nbufs]);
    for (int b = 0; b < nbufs && ok; b++) {
        if (!cnt[b]) continue;
        const Bind w = {&p->bufs[b], 0}, in = {&t->scratch, t->xo_eq}, td = {&p->tab, (VkDeviceSize)(2 * XP_ROWS + rb[b]) * 8};
        const Bind o = {&t->scratch, t->xo_eo + (size_t)rb[b] * D * 4};
        ok = mvx(v, &p->tm[layer * 3 + 2], w, td, D, o, in, cnt[b], D, (uint32_t)nbf);
    }
    compute_barrier(v);
    const uint32_t cc[16] = {(uint32_t)D, (uint32_t)t->xo_eo, (uint32_t)(t->o_ge + (size_t)t->cur_t0 * D * 4), (uint32_t)K,
                             (uint32_t)t->in_c};
    ok = ok && tok_run(v, T_COMBINE, cc, (uint32_t)(D / 256), (uint32_t)t->cur_n);
    ok = kick(v) && ok;
    v->ln = &v->lanes[0];
    if (!ok) {
        v->broken = 1;
        return 0;
    }
    t->ge_on = 1;
    return 1;
}

void vk_rebalance(Vulkan *v, int max_uploads) {
    VExp *p = xp_of(v);
    if (!p || v->lanes[0].recording || v->lanes[0].inflight) return;   /* the last uploads have not run yet */
    const int ne = p->nl * p->E;
    for (int i = 0; i < ne; i++) p->score[i] *= XP_DECAY;
    int up = p->n_free ? XP_STAGE : max_uploads;
    if (up > XP_STAGE) up = XP_STAGE;
    for (int u = 0; u < up; u++) {
        int best = -1;   /* the best scored expert not in VRAM */
        for (int i = 0; i < ne; i++)
            if (p->slot_of[i] < 0 && p->score[i] > 0.5f && (best < 0 || p->score[i] > p->score[best])) best = i;
        if (best < 0) break;
        int slot = -1;
        if (p->n_free) {
            for (slot = 0; p->owner[slot] >= 0; slot++) {}
            p->n_free--;
        } else {
            for (int s2 = 0; s2 < p->n_slots; s2++)
                if (slot < 0 || p->score[p->owner[s2]] < p->score[p->owner[slot]]) slot = s2;
            if (p->score[best] < 1.5f * p->score[p->owner[slot]] + 0.5f) break;   /* not worth a swap */
            p->slot_of[p->owner[slot]] = -1;
        }
        const int l = best / p->E, e = best % p->E;
        const Layer *L = &v->dense->m->layers[l];
        const Mat *ws[3] = {&L->gate_exps, &L->up_exps, &L->down_exps};
        uint8_t *dst = p->stage.map + (size_t)u * p->slot_bytes;
        for (int k = 0; k < 3; k++) {
            VMat x = p->tm[l * 3 + k];
            x.host = ws[k]->data + (size_t)e * ws[k]->expert_stride;
            pack_rows(&x, 0, x.rows, dst + p->toff[l * 3 + k]);
        }
        copy(v, &p->bufs[slot / p->per_buf], (VkDeviceSize)(slot % p->per_buf) * p->slot_bytes, &p->stage,
             (VkDeviceSize)u * p->slot_bytes, p->slot_bytes);
        p->owner[slot] = best;
        p->slot_of[best] = slot;
    }
    if (v->ln->recording) barrier(v, PS_TRANSFER, AC_TRANSFER_WRITE, PS_COMPUTE, AC_SHADER_READ);
}

void vk_dense_set_moe(Vulkan *v, VkMoeFn fn, void *arg) {
    if (v && v->dense) {
        v->dense->moe = fn;
        v->dense->moe_arg = arg;
    }
}

int vk_dense_forward(Vulkan *v, const float *x, int n, const float *rope_cs, const float *rope_swa, int pos0,
                     float *logits, float *x_last) {
    VDense *t = v ? v->dense : NULL;
    if (!t || v->broken || n < 1 || n > t->B || pos0 + n > t->n_ctx || (t->hybrid && !t->moe)) return 0;
    static int passes;   /* PITHOS_VK_FAIL=N: the GPU "fails" at pass N, to test the CPU fallback */
    if (getenv("PITHOS_VK_FAIL") && ++passes == atoi(getenv("PITHOS_VK_FAIL"))) {
        v->broken = 1;
        fprintf(stderr, "Vulkan: simulated GPU failure (PITHOS_VK_FAIL)\n");
        return 0;
    }
    if (t->hybrid) return hybrid_forward(v, x, n, rope_cs, pos0, logits, x_last);
    const Model *m = t->m;
    const int D = t->D, hd = t->hd, kvd = t->kvd, qkvw = D + 2 * kvd;
    const uint32_t nb = (uint32_t)(D / QK_K), un = (uint32_t)n;
    memcpy(t->in.map, x, sizeof(float) * (size_t)n * D);
    memcpy(t->in.map + 4 * (size_t)n * D, rope_cs, sizeof(float) * (size_t)n * hd);
    if (rope_swa) memcpy(t->in.map + 4 * (size_t)n * (D + hd), rope_swa, sizeof(float) * (size_t)n * hd);

    int ok = tok_resid(v, 0, NONE_OFF, t->layers[0].attn_norm, n);
    for (int l = 0; l < m->n_layer && ok; l++) {
        const Layer *L = &m->layers[l];
        const VLayer *X = &t->layers[l];
        if (L->is_attn) {
            ok = tok_mm(v, X->wq, t->o_hq, t->o_big, n, qkvw) && tok_mm(v, X->wk, t->o_hq, t->o_big + 4 * (size_t)D, n, qkvw) &&
                 tok_mm(v, X->wv, t->o_hq, t->o_big + 4 * (size_t)(D + kvd), n, qkvw) &&
                 (!X->gate || tok_mm(v, X->gate, t->o_hq, t->o_gate, n, t->nh));
            compute_barrier(v);
            const size_t kbase = (size_t)L->attn_idx * (size_t)t->n_ctx * (size_t)kvd * 2;
            const size_t krow = kbase + (size_t)pos0 * (size_t)kvd * 2;
            const uint32_t rope = (uint32_t)(4 * (size_t)n * (L->is_swa && rope_swa ? D + hd : D));
            const uint32_t cp[16] = {(uint32_t)hd, (uint32_t)t->o_big, (uint32_t)D, (uint32_t)kvd, X->q_norm, X->k_norm, rope,
                                     fbits(m->eps), (uint32_t)krow, (uint32_t)(t->kv_half + krow), (uint32_t)t->nh};
            ok = ok && tok_run(v, T_PREP, cp, (uint32_t)(t->nh + t->nkv), un);
            const uint32_t gate = X->gate ? (uint32_t)t->o_gate : NONE_OFF, scale = fbits(1.f / sqrtf((float)hd));
            if (n == 1) {
                const int npos = pos0 + 1, j0 = L->is_swa && npos > m->swa_window ? npos - m->swa_window : 0;
                const uint32_t ca[16] = {(uint32_t)hd, (uint32_t)t->o_big, gate, (uint32_t)t->o_tmp,
                                         (uint32_t)kbase, (uint32_t)(t->kv_half + kbase), (uint32_t)kvd, (uint32_t)t->o_sc,
                                         (uint32_t)j0, (uint32_t)npos, (uint32_t)(t->nh / t->nkv), (uint32_t)t->n_ctx, scale};
                ok = ok && tok_run(v, T_ATTN, ca, (uint32_t)t->nh, 1);
            } else {
                const uint32_t ca[16] = {(uint32_t)hd, (uint32_t)t->o_big, gate, (uint32_t)t->o_tmp,
                                         (uint32_t)kbase, (uint32_t)(t->kv_half + kbase), (uint32_t)kvd, 0,
                                         (uint32_t)pos0, (uint32_t)(L->is_swa && m->swa_window > 0 ? m->swa_window : 0),
                                         (uint32_t)(t->nh / t->nkv), (uint32_t)t->nh, scale, (uint32_t)D};
                ok = ok && tok_run(v, T_ATTN_CHUNK, ca, (uint32_t)t->nh, un);
            }
            const uint32_t cq[16] = {(uint32_t)t->o_tmp, (uint32_t)t->o_hq, (uint32_t)D * 4, nb * Q8_BYTES};
            ok = ok && tok_run(v, T_QUANT, cq, nb, un) && tok_mm(v, X->wo, t->o_hq, t->o_out, n, D);
        } else {
            ok = tok_mm(v, X->in_proj, t->o_hq, t->o_big, n, 3 * D);
            compute_barrier(v);
            const uint32_t state = (uint32_t)((size_t)L->conv_idx * (size_t)(t->lc - 1) * (size_t)D * 4);
            const uint32_t cc[16] = {(uint32_t)D, (uint32_t)t->o_big, (uint32_t)t->o_hq, (uint32_t)t->lc, X->conv_w, state};
            if (n == 1) {
                ok = ok && tok_run(v, T_CONV, cc, nb, 1);
            } else {
                const uint32_t cs[16] = {(uint32_t)D, (uint32_t)t->o_big, un, (uint32_t)t->lc, 0, state};
                ok = ok && tok_run(v, T_CONV_CHUNK, cc, nb, un) && tok_run(v, T_CONV_STATE, cs, nb, 1);
            }
            ok = ok && tok_mm(v, X->out_proj, t->o_hq, t->o_out, n, D);
        }
        compute_barrier(v);
        ok = ok && tok_resid(v, 1, X->post_op, X->ffn_norm, n);
        const int F = X->w_gate->rows;
        ok = ok && tok_mm(v, X->w_gate, t->o_hq, t->o_fg, n, F) && tok_mm(v, X->w_up, t->o_hq, t->o_fu, n, F);
        compute_barrier(v);
        const uint32_t cs[16] = {(uint32_t)t->o_fg, (uint32_t)t->o_fu, (uint32_t)t->o_fq, (uint32_t)F};
        ok = ok && tok_run(v, T_SILU, cs, (uint32_t)(F / QK_K), un) && tok_mm(v, X->w_down, t->o_fq, t->o_out, n, D);
        compute_barrier(v);
        ok = ok && tok_resid(v, 1, X->post_ffn, l + 1 < m->n_layer ? t->layers[l + 1].attn_norm : t->final_norm, n);
    }
    return finish_pass(v, ok, n, logits, x_last);
}

void vk_dense_kv(Vulkan *v, uint16_t *kc, uint16_t *vc, int r0, int r1, int to_gpu) {
    VDense *t = v ? v->dense : NULL;
    if (!t || v->broken || r1 <= r0 || !t->n_attn) return;
    const size_t row = (size_t)t->kvd * 2, bytes = (size_t)(r1 - r0) * row, total = 2 * (size_t)t->n_attn * bytes;
    VBuf *st = to_gpu ? &v->up : &v->rb;
    if (!vbuf_reserve(v, st, total, to_gpu ? HOST_UPLOAD : HOST_READBACK)) return;
    for (int a = 0; a < t->n_attn; a++)
        for (int h = 0; h < 2; h++) {
            uint16_t *host = h ? vc : kc;
            const size_t off = ((size_t)a * (size_t)t->n_ctx + (size_t)r0) * row, gpu = (h ? t->kv_half : 0) + off;
            const size_t so = (2 * (size_t)a + (size_t)h) * bytes;
            if (to_gpu) {
                memcpy(st->map + so, (uint8_t *)host + off, bytes);
                copy(v, &t->kv, gpu, st, so, bytes);
            } else {
                copy(v, st, so, &t->kv, gpu, bytes);
            }
        }
    if (!to_gpu) barrier(v, PS_TRANSFER, AC_TRANSFER_WRITE, PS_HOST, AC_HOST_READ);
    if (!submit(v)) { v->broken = 1; return; }
    if (!to_gpu)
        for (int a = 0; a < t->n_attn; a++)
            for (int h = 0; h < 2; h++) {
                const size_t off = ((size_t)a * (size_t)t->n_ctx + (size_t)r0) * row;
                memcpy((uint8_t *)(h ? vc : kc) + off, st->map + (2 * (size_t)a + (size_t)h) * bytes, bytes);
            }
}

void vk_dense_conv(Vulkan *v, float *conv, int to_gpu) {
    VDense *t = v ? v->dense : NULL;
    if (!t || v->broken || !t->cs_bytes) return;
    const size_t bytes = t->cs_bytes;
    VBuf *st = to_gpu ? &v->up : &v->rb;
    if (!vbuf_reserve(v, st, bytes, to_gpu ? HOST_UPLOAD : HOST_READBACK)) return;
    if (to_gpu) {
        memcpy(st->map, conv, bytes);
        copy(v, &t->cs, 0, st, 0, bytes);
    } else {
        copy(v, st, 0, &t->cs, 0, bytes);
        barrier(v, PS_TRANSFER, AC_TRANSFER_WRITE, PS_HOST, AC_HOST_READ);
    }
    if (!submit(v)) { v->broken = 1; return; }
    if (!to_gpu) memcpy(conv, st->map, bytes);
}

#endif
