#include <cstdint>
#include <iostream>
#include <fstream>
#include <optional>

#include <vector>

#include <vulkan/vulkan_core.h>

#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>

#include <VkBootstrap.h>

namespace {

struct ShaderConstants {
	float width;
	float height;
	float time;
};

constexpr uint32_t window_default_width = 1280;
constexpr uint32_t window_default_height = 720;
constexpr char window_title[] = "Veekay";

constexpr uint32_t max_frames_in_flight = 2;

GLFWwindow* window;

VkInstance vk_instance;
VkDebugUtilsMessengerEXT vk_debug_messenger;
VkPhysicalDevice vk_physical_device;
VkDevice vk_device;
VkSurfaceKHR vk_surface;

VkSwapchainKHR vk_swapchain;
VkFormat vk_swapchain_format;
std::vector<VkImage> vk_swapchain_images;
std::vector<VkImageView> vk_swapchain_image_views;

VkQueue vk_graphics_queue;
uint32_t vk_graphics_queue_family;

VkCommandPool vk_command_pool;
std::vector<VkCommandBuffer> vk_command_buffers;

VkRenderPass vk_render_pass;
std::vector<VkFramebuffer> vk_framebuffers;

std::vector<VkSemaphore> vk_render_semaphores;
std::vector<VkSemaphore> vk_present_semaphores;
std::vector<VkFence> vk_in_flight_fences;
uint32_t vk_current_frame;

VkShaderModule vk_vertex_shader_module;
VkShaderModule vk_fragment_shader_module;
VkPipelineLayout vk_pipeline_layout;
VkPipeline vk_pipeline;

std::optional<VkShaderModule> loadShaderModule(const char* path) {
	std::ifstream file(path, std::ios::binary | std::ios::ate);
	size_t size = file.tellg();
	std::vector<uint32_t> buffer(size / sizeof(uint32_t));
	file.seekg(0);
	file.read(reinterpret_cast<char*>(buffer.data()), size);
	file.close();

	VkShaderModuleCreateInfo info{
		.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
		.codeSize = size,
		.pCode = buffer.data(),
	};

	VkShaderModule result;
	if (vkCreateShaderModule(vk_device, &info, nullptr, &result) != VK_SUCCESS) {
		return std::nullopt;
	}

	return std::make_optional<VkShaderModule>(result);
}

} // namespace

int main() {
	if (!glfwInit()) {
		std::cerr << "Failed to initialize GLFW\n";
		return 1;
	}

	glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
	glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);

	window = glfwCreateWindow(window_default_width, window_default_height,
	                          window_title, nullptr, nullptr);
	if (!window) {
		std::cerr << "Failed to create GLFW window\n";
		return 1;
	}

	{ // NOTE: Initialize Vulkan: grab device and create swapchain
		vkb::InstanceBuilder instance_builder;

		auto builder_result = instance_builder.require_api_version(1, 2, 0)
		                                      .request_validation_layers()
		                                      .use_default_debug_messenger()
		                                      .build();
		if (!builder_result) {
			std::cerr << builder_result.error().message() << '\n';
			return 1;
		}

		auto instance = builder_result.value();

		vk_instance = instance.instance;
		vk_debug_messenger = instance.debug_messenger;

		if (glfwCreateWindowSurface(vk_instance, window, nullptr, &vk_surface) != VK_SUCCESS) {
			const char* message;
			glfwGetError(&message);
			std::cerr << message << '\n';
			return 1;
		}

		vkb::PhysicalDeviceSelector physical_device_selector(instance);

		auto selector_result = physical_device_selector.set_surface(vk_surface)
		                                               .select();
		if (!selector_result) {
			std::cerr << selector_result.error().message() << '\n';
			return 1;
		}

		auto physical_device = selector_result.value();

		{
			vkb::DeviceBuilder device_builder(physical_device);

			auto result = device_builder.build();

			if (!result) {
				std::cerr << result.error().message() << '\n';
				return 1;
			}

			auto device = result.value();

			vk_device = device.device;
			vk_physical_device = device.physical_device;

			auto queue_type = vkb::QueueType::graphics;
			
			vk_graphics_queue = device.get_queue(queue_type).value();
			vk_graphics_queue_family = device.get_queue_index(queue_type).value();
		}

		vkb::SwapchainBuilder swapchain_builder(vk_physical_device, vk_device, vk_surface);

		vk_swapchain_format = VK_FORMAT_B8G8R8A8_UNORM;

		VkSurfaceFormatKHR surface_format{
			.format = vk_swapchain_format,
			.colorSpace = VK_COLOR_SPACE_SRGB_NONLINEAR_KHR,
		};

		auto swapchain_result = swapchain_builder.set_desired_format(surface_format)
		                                         .set_desired_present_mode(VK_PRESENT_MODE_FIFO_KHR)
		                                         .set_desired_extent(window_default_width, window_default_height)
		                                         .add_image_usage_flags(VK_IMAGE_USAGE_TRANSFER_DST_BIT)
		                                         .build();

		if (!swapchain_result) {
			std::cerr << swapchain_result.error().message() << '\n';
			return 1;
		}

		auto swapchain = swapchain_result.value();

		vk_swapchain = swapchain.swapchain;
		vk_swapchain_images = swapchain.get_images().value();
		vk_swapchain_image_views = swapchain.get_image_views().value();
	}

	{ // NOTE: Create render pass
		VkAttachmentDescription attachment{
			.format = vk_swapchain_format,

			.samples = VK_SAMPLE_COUNT_1_BIT,

			.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
			.storeOp = VK_ATTACHMENT_STORE_OP_STORE,

			.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
			.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,

			.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
			.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
		};

		VkAttachmentReference ref{
			.attachment = 0,
			.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
		};

		VkSubpassDescription subpass{
			.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS,
			.colorAttachmentCount = 1,
			.pColorAttachments = &ref,
		};

		VkRenderPassCreateInfo info{
			.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO,

			.attachmentCount = 1,
			.pAttachments = &attachment,

			.subpassCount = 1,
			.pSubpasses = &subpass,
		};

		if (vkCreateRenderPass(vk_device, &info, nullptr, &vk_render_pass) != VK_SUCCESS) {
			std::cerr << "Failed to create render pass\n";
			return 1;
		}
	}

	{ // NOTE: Create framebuffer objects from swapchain images
		VkFramebufferCreateInfo info{
			.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,

			.renderPass = vk_render_pass,
			.attachmentCount = 1,

			.width = window_default_width,
			.height = window_default_height,
			.layers = 1,
		};

		const uint32_t n = vk_swapchain_images.size();

		vk_framebuffers.resize(n);

		for (uint32_t i = 0; i < n; ++i) {
			info.pAttachments = &vk_swapchain_image_views[i];
			if (vkCreateFramebuffer(vk_device, &info, nullptr, &vk_framebuffers[i]) != VK_SUCCESS) {
				std::cerr << "Failed to create Vulkan framebuffer " << i << '\n';
				return 1;
			}
		}
	}

	{ // NOTE: Create sync primitives
		VkFenceCreateInfo fence_info{
			.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
			.flags = VK_FENCE_CREATE_SIGNALED_BIT,
		};

		VkSemaphoreCreateInfo sem_info{
			.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO,
		};

		vk_present_semaphores.resize(vk_swapchain_images.size());

		for (uint32_t i = 0, n = vk_swapchain_images.size(); i < n; ++i) {
			vkCreateSemaphore(vk_device, &sem_info, nullptr, &vk_present_semaphores[i]);
		}

		vk_render_semaphores.resize(max_frames_in_flight);
		vk_in_flight_fences.resize(max_frames_in_flight);

		for (uint32_t i = 0; i < max_frames_in_flight; ++i) {
			vkCreateSemaphore(vk_device, &sem_info, nullptr, &vk_render_semaphores[i]);
			vkCreateFence(vk_device, &fence_info, nullptr, &vk_in_flight_fences[i]);
		}
	}

	{ // NOTE: Create command pool from graphics queue
		VkCommandPoolCreateInfo info{
			.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
			.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
			.queueFamilyIndex = vk_graphics_queue_family,
		};

		if (vkCreateCommandPool(vk_device, &info, nullptr, &vk_command_pool) != VK_SUCCESS) {
			std::cerr << "Failed to create Vulkan command pool\n";
			return 1;
		}
	}

	{ // NOTE: Allocate command buffers
		vk_command_buffers.resize(vk_framebuffers.size());
		
		VkCommandBufferAllocateInfo info{
			.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
			.commandPool = vk_command_pool,
			.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
			.commandBufferCount = static_cast<uint32_t>(vk_command_buffers.size()),
		};

		if (vkAllocateCommandBuffers(vk_device, &info, vk_command_buffers.data()) != VK_SUCCESS) {
			std::cerr << "Failed to allocate Vulkan command buffers\n";
			return 1;
		}
	}

	{ // NOTE: Build graphics pipeline
		auto vertex_shader = loadShaderModule("./shaders/shader.vert.spv");
		if (!vertex_shader) {
			std::cerr << "Failed to load Vulkan vertex shader from file\n";
			return 1;
		}

		auto fragment_shader = loadShaderModule("./shaders/shader.frag.spv");
		if (!fragment_shader) {
			std::cerr << "Failed to load Vulkan fragment shader from file\n";
			return 1;
		}

		vk_vertex_shader_module = vertex_shader.value();
		vk_fragment_shader_module = fragment_shader.value();

		VkPipelineShaderStageCreateInfo stage_infos[2];

		stage_infos[0] = VkPipelineShaderStageCreateInfo{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
			.stage = VK_SHADER_STAGE_VERTEX_BIT,
			.module = vk_vertex_shader_module,
			.pName = "main",
		};

		stage_infos[1] = VkPipelineShaderStageCreateInfo{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
			.stage = VK_SHADER_STAGE_FRAGMENT_BIT,
			.module = vk_fragment_shader_module,
			.pName = "main",
		};

		VkPipelineVertexInputStateCreateInfo input_state_info{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
		};

		VkPipelineInputAssemblyStateCreateInfo assembly_state_info{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
			.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP,
		};

		VkPipelineRasterizationStateCreateInfo raster_info{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
			.polygonMode = VK_POLYGON_MODE_FILL,
			.cullMode = VK_CULL_MODE_NONE,
			.frontFace = VK_FRONT_FACE_CLOCKWISE,
			.lineWidth = 1.0f,
		};

		VkPipelineMultisampleStateCreateInfo sample_info{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
			.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT,
			.sampleShadingEnable = false,
			.minSampleShading = 1.0f,
		};

		VkPipelineColorBlendAttachmentState attachment_info{
			.colorWriteMask = VK_COLOR_COMPONENT_R_BIT |
			                  VK_COLOR_COMPONENT_G_BIT |
			                  VK_COLOR_COMPONENT_B_BIT |
			                  VK_COLOR_COMPONENT_A_BIT,
		};

		VkViewport viewport{
			.x = 0.0f,
			.y = 0.0f,
			.width = static_cast<float>(window_default_width),
			.height = static_cast<float>(window_default_height),
			.minDepth = 0.0f,
			.maxDepth = 1.0f,
		};

		VkRect2D scissor{
			.offset = {0, 0},
			.extent = {window_default_width, window_default_height},
		};

		VkPipelineViewportStateCreateInfo viewport_info{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,

			.viewportCount = 1,
			.pViewports = &viewport,

			.scissorCount = 1,
			.pScissors = &scissor,
		};

		VkPipelineColorBlendStateCreateInfo blend_info{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,

			.logicOpEnable = false,
			.logicOp = VK_LOGIC_OP_COPY,

			.attachmentCount = 1,
			.pAttachments = &attachment_info
		};

		VkPushConstantRange push_consts{
			.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
			.size = sizeof(ShaderConstants),
		};

		VkPipelineLayoutCreateInfo layout_info{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,

			.pushConstantRangeCount = 1,
			.pPushConstantRanges = &push_consts,
		};

		if (vkCreatePipelineLayout(vk_device, &layout_info, nullptr, &vk_pipeline_layout) != VK_SUCCESS) {
			std::cerr << "Failed to create Vulkan pipeline layout\n";
			return 1;
		}
		
		VkGraphicsPipelineCreateInfo info{
			.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
			.stageCount = 2,
			.pStages = stage_infos,
			.pVertexInputState = &input_state_info,
			.pInputAssemblyState = &assembly_state_info,
			.pViewportState = &viewport_info,
			.pRasterizationState = &raster_info,
			.pMultisampleState = &sample_info,
			.pColorBlendState = &blend_info,
			.layout = vk_pipeline_layout,
			.renderPass = vk_render_pass,
		};

		if (vkCreateGraphicsPipelines(vk_device, nullptr, 1, &info, nullptr, &vk_pipeline) != VK_SUCCESS) {
			std::cerr << "Failed to create Vulkan pipeline\n";
			return 1;
		}
	}

	while (!glfwWindowShouldClose(window)) {
		glfwPollEvents();
		float time = static_cast<float>(glfwGetTime());

		// NOTE: Wait until the previous frame finishes
		vkWaitForFences(vk_device, 1, &vk_in_flight_fences[vk_current_frame], true, UINT64_MAX);
		vkResetFences(vk_device, 1, &vk_in_flight_fences[vk_current_frame]);

		// NOTE: Get current swapchain framebuffer index
		uint32_t swapchain_image_index = 0;
		vkAcquireNextImageKHR(vk_device, vk_swapchain, UINT64_MAX,
		                      vk_render_semaphores[vk_current_frame],
		                      nullptr, &swapchain_image_index);

		// NOTE: Grab current frame command buffer
		VkCommandBuffer cmd = vk_command_buffers[swapchain_image_index];
		vkResetCommandBuffer(cmd, 0);

		{ // NOTE: Start recording rendering commands
			VkCommandBufferBeginInfo info{
				.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
				.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
			};

			vkBeginCommandBuffer(cmd, &info);
		}

		{ // NOTE: Use current swapchain framebuffer and clear it
			VkClearValue clear_values{
				.color = {{0.1f, 0.1f, 0.1f, 1.0f}},
			};

			VkRenderPassBeginInfo info{
				.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
				.renderPass = vk_render_pass,
				.framebuffer = vk_framebuffers[swapchain_image_index],
				.renderArea = {
					.extent = {window_default_width, window_default_height},
				},
				.clearValueCount = 1,
				.pClearValues = &clear_values,
			};

			vkCmdBeginRenderPass(cmd, &info, VK_SUBPASS_CONTENTS_INLINE);
		}

		{ // NOTE: Draw!
			vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, vk_pipeline);

			ShaderConstants consts{
				.width = window_default_width,
				.height = window_default_height,
				.time = time,
			};
			vkCmdPushConstants(cmd, vk_pipeline_layout,
			                   VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
			                   0, sizeof(ShaderConstants), &consts);
			
			vkCmdDraw(cmd, 4, 1, 0, 0);
		}

		// NOTE: Stop recording rendering commands
		vkCmdEndRenderPass(cmd);
		vkEndCommandBuffer(cmd);

		{ // NOTE: Submit commands to graphics queue
			VkPipelineStageFlags wait_stage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;

			VkSubmitInfo info{
				.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
				.waitSemaphoreCount = 1,
				.pWaitSemaphores = &vk_render_semaphores[vk_current_frame],
				.pWaitDstStageMask = &wait_stage,
				.commandBufferCount = 1,
				.pCommandBuffers = &cmd,
				.signalSemaphoreCount = 1,
				.pSignalSemaphores = &vk_present_semaphores[swapchain_image_index],
			};

			vkQueueSubmit(vk_graphics_queue, 1, &info, vk_in_flight_fences[vk_current_frame]);
		}

		{ // NOTE: Present renderer frame
			VkPresentInfoKHR info{
				.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR,
				.waitSemaphoreCount = 1,
				.pWaitSemaphores = &vk_present_semaphores[swapchain_image_index],
				.swapchainCount = 1,
				.pSwapchains = &vk_swapchain,
				.pImageIndices = &swapchain_image_index,
			};

			vkQueuePresentKHR(vk_graphics_queue, &info);

			vk_current_frame = (vk_current_frame + 1) % max_frames_in_flight;
		}
	}

	vkDeviceWaitIdle(vk_device);

	for (uint32_t i = 0, n = vk_swapchain_images.size(); i < n; ++i) {
		vkDestroySemaphore(vk_device, vk_present_semaphores[i], nullptr);
	}

	for (uint32_t i = 0; i < max_frames_in_flight; ++i) {
		vkDestroySemaphore(vk_device, vk_render_semaphores[i], nullptr);
		vkDestroyFence(vk_device, vk_in_flight_fences[i], nullptr);
	}

	vkDestroyCommandPool(vk_device, vk_command_pool, nullptr);

	vkDestroyPipeline(vk_device, vk_pipeline, nullptr);
	vkDestroyPipelineLayout(vk_device, vk_pipeline_layout, nullptr);

	vkDestroyShaderModule(vk_device, vk_fragment_shader_module, nullptr);
	vkDestroyShaderModule(vk_device, vk_vertex_shader_module, nullptr);
	
	vkDestroyRenderPass(vk_device, vk_render_pass, nullptr);

	for (uint32_t i = 0, n = vk_framebuffers.size(); i < n; ++i) {
		vkDestroyFramebuffer(vk_device, vk_framebuffers[i], nullptr);
		vkDestroyImageView(vk_device, vk_swapchain_image_views[i], nullptr);
	}
	
	vkDestroySwapchainKHR(vk_device, vk_swapchain, nullptr);
	vkDestroyDevice(vk_device, nullptr);
	vkDestroySurfaceKHR(vk_instance, vk_surface, nullptr);
	vkb::destroy_debug_utils_messenger(vk_instance, vk_debug_messenger);
	vkDestroyInstance(vk_instance, nullptr);

	glfwDestroyWindow(window);
	glfwTerminate();
	
	return 0;
}
