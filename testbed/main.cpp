#include <cstdint>
#include <climits>
#include <cstring>
#include <vector>
#include <iostream>
#include <fstream>
#include <cmath>
#include <algorithm>

#include <veekay/veekay.hpp>

#include <vulkan/vulkan_core.h>
#include <imgui.h>
#include <lodepng.h>

namespace {

constexpr uint32_t max_models = 1024;
constexpr uint32_t max_point_lights = 16;
constexpr uint32_t max_spot_lights  = 16;

struct Vertex {
	veekay::vec3 position;
	veekay::vec3 normal;
	veekay::vec2 uv;
	// NOTE: You can add more attributes
};

struct SceneUniforms {
	veekay::mat4 view_projection;
	veekay::vec3 view_position; float _pad0;

	veekay::vec3 ambient_light_intensity; float _pad1;

	veekay::vec3 sun_light_direction; float _pad2;
	veekay::vec3 sun_light_color; float _pad3;
	uint32_t point_lights_count;
    uint32_t spot_lights_count;
    uint32_t _pad4[2];
};

struct ModelUniforms {
	veekay::mat4 model;
	veekay::vec3 albedo_color; float _pad0;
	veekay::vec3 specular_color; float _pad1;
	float shininess; float _pad2[3];
};

struct Mesh {
	veekay::graphics::Buffer* vertex_buffer;
	veekay::graphics::Buffer* index_buffer;
	uint32_t indices;
};

struct Transform {
	veekay::vec3 position = {};
	veekay::vec3 scale = {1.0f, 1.0f, 1.0f};
	veekay::vec3 rotation = {};

	// NOTE: Model matrix (translation, rotation and scaling)
	veekay::mat4 matrix() const;
};

struct Model {
	Mesh mesh;
	Transform transform;
	veekay::vec3 albedo_color;
	veekay::vec3 specular_color;
    float shininess;
};

struct Camera {
	constexpr static float default_fov = 60.0f;
	constexpr static float default_near_plane = 0.01f;
	constexpr static float default_far_plane = 100.0f;

	veekay::vec3 position = {};
	veekay::vec3 rotation = {};

	float fov = default_fov;
	float near_plane = default_near_plane;
	float far_plane = default_far_plane;

	// NOTE: View matrix of camera (inverse of a transform)
	veekay::mat4 view() const;

	// NOTE: Look-at view matrix
	veekay::mat4 lookat() const;

	// NOTE: View and projection composition
	veekay::mat4 view_projection(float aspect_ratio) const;
};

struct PointLight {
    veekay::vec3 position;
    float radius;           // для затухания и удобства
    veekay::vec3 color;
    float _pad0;
};
struct SpotLight {
    veekay::vec3 position;
    float radius;

    veekay::vec3 direction;
    float angle;            // cos(угла) конуса

    veekay::vec3 color;
    float _pad0;
};

// NOTE: Scene objects
inline namespace {
	bool g_use_lookat = false;
	float g_mouse_sensitivity = 0.12f * float(M_PI) / 180.0f;
	veekay::vec3 g_ambient = { 0.15f, 0.15f, 0.15f };
	veekay::vec3 g_sun_dir = veekay::vec3::normalized({ -0.3f, 1.0f, -0.2f });
	veekay::vec3 g_sun_color = {0.5f, 0.5f, 0.5f};

	Camera camera{
		.position = {0.0f, -0.5f, -3.0f}
	};

	PointLight pl {
		.position = { 0.0f, -2.0f, 0.0f },
		.radius   = 5.0f,
		.color    = { 1.0f, 1.0f, 1.0f }
	};

	std::vector<Model> models;
	std::vector<PointLight> point_lights;
	std::vector<SpotLight>  spot_lights;
}

// NOTE: Vulkan objects
inline namespace {
	VkShaderModule vertex_shader_module;
	VkShaderModule fragment_shader_module;

	VkDescriptorPool descriptor_pool;
	VkDescriptorSetLayout descriptor_set_layout;
	VkDescriptorSet descriptor_set;

	VkPipelineLayout pipeline_layout;
	VkPipeline pipeline;

	veekay::graphics::Buffer* scene_uniforms_buffer;
	veekay::graphics::Buffer* model_uniforms_buffer;

	veekay::graphics::Buffer* point_lights_buffer;
	veekay::graphics::Buffer* spot_lights_buffer;

	Mesh plane_mesh;
	Mesh cube_mesh;

	veekay::graphics::Texture* missing_texture;
	VkSampler missing_texture_sampler;

	veekay::graphics::Texture* texture;
	VkSampler texture_sampler;
}

float toRadians(float degrees) {
	return degrees * float(M_PI) / 180.0f;
}

static constexpr veekay::vec3 WORLD_UP{0.0f, -1.0f, 0.0f};

// Базис из yaw/pitch (в градусах), без roll — «игровая» камера
static void yawPitchBasis(float yaw, float pitch,
                          veekay::vec3& right,
                          veekay::vec3& up,
                          veekay::vec3& front)
{
    // фронт: -Z при yaw=0, pitch=0
    front = veekay::vec3::normalized({
        std::cosf(pitch) * std::sinf(yaw), // x
        std::sinf(pitch),                  // y
        -std::cosf(pitch) * std::cosf(yaw) // z
    });
    right = veekay::vec3::normalized(veekay::vec3::cross(front, WORLD_UP));
    up    = veekay::vec3::cross(right, front);
}

// Классический lookAt (row-major)
veekay::mat4 Camera::lookat() const {
    using veekay::vec3;
    using veekay::mat4;

    // 1) Строим базис камеры из rotation (в радианах)
    vec3 right, up, front;
    yawPitchBasis(rotation.y, rotation.x, right, up, front);

    vec3 eye    = position;
    vec3 center = { eye.x + front.x,
                    eye.y + front.y,
                    eye.z + front.z };

    // 2) Классический lookAt (row-major), но up берём как -up,
    //    т.к. у нас WORLD_UP = (0, -1, 0) и так камера "выпрямлена"
    vec3 up_vec = -up;

    vec3 f = vec3::normalized({ center.x - eye.x,
                                center.y - eye.y,
                                center.z - eye.z });   // forward
    vec3 r = vec3::normalized(vec3::cross(f, up_vec)); // right
    vec3 u = vec3::cross(r, f);                        // up

    mat4 m{};
    m[0][0] = r.x; m[0][1] = u.x; m[0][2] = -f.x; m[0][3] = 0.0f;
    m[1][0] = r.y; m[1][1] = u.y; m[1][2] = -f.y; m[1][3] = 0.0f;
    m[2][0] = r.z; m[2][1] = u.z; m[2][2] = -f.z; m[2][3] = 0.0f;
    m[3][0] = -vec3::dot(r, eye);
    m[3][1] = -vec3::dot(u, eye);
    m[3][2] =  vec3::dot(f, eye);
    m[3][3] =  1.0f;

    return m;
}

veekay::mat4 Transform::matrix() const {
    using veekay::vec3; using veekay::mat4;
    mat4 S = mat4::scaling(scale);

    // Порядок при row-major с умножением справа налево: R = Rz * Rx * Ry
    mat4 Rz = mat4::rotation(vec3{0,0,1}, rotation.z);
    mat4 Rx = mat4::rotation(vec3{1,0,0}, rotation.x);
    mat4 Ry = mat4::rotation(vec3{0,1,0}, rotation.y);
    mat4 R  = Rz * Rx * Ry;

    mat4 T = mat4::translation(position);

    return S * R * T;
}

veekay::mat4 Camera::view() const {
    veekay::vec3 right, up, front;
    // rotation.x, rotation.y — уже в радианах
    yawPitchBasis(rotation.y, rotation.x, right, up, front);

    // Матрица вида (как на слайдах): смена базиса + обратная трансляция.
    veekay::mat4 m{};
    m[0][0] = -right.x; m[0][1] = -up.x; m[0][2] = -front.x; m[0][3] = 0.0f;
    m[1][0] = -right.y; m[1][1] = -up.y; m[1][2] = -front.y; m[1][3] = 0.0f;
    m[2][0] = -right.z; m[2][1] = -up.z; m[2][2] = -front.z; m[2][3] = 0.0f;
    m[3][0] =  veekay::vec3::dot(right,  position);
    m[3][1] =  veekay::vec3::dot(up,     position);
    m[3][2] =  veekay::vec3::dot(front,  position);
    m[3][3] =  1.0f;
    return m;
}

veekay::mat4 Camera::view_projection(float aspect_ratio) const {
	auto projection = veekay::mat4::projection(fov, aspect_ratio, near_plane, far_plane);

	veekay::mat4 view_matrix = g_use_lookat ? lookat() : view();

	return view_matrix * projection;
}

// NOTE: Loads shader byte code from file
// NOTE: Your shaders are compiled via CMake with this code too, look it up
VkShaderModule loadShaderModule(const char* path) {
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
	if (vkCreateShaderModule(veekay::app.vk_device, &
	                         info, nullptr, &result) != VK_SUCCESS) {
		return nullptr;
	}

	return result;
}

void initialize(VkCommandBuffer cmd) {
	VkDevice& device = veekay::app.vk_device;
	VkPhysicalDevice& physical_device = veekay::app.vk_physical_device;

	{ // NOTE: Build graphics pipeline
		vertex_shader_module = loadShaderModule("./shaders/shader.vert.spv");
		if (!vertex_shader_module) {
			std::cerr << "Failed to load Vulkan vertex shader from file\n";
			veekay::app.running = false;
			return;
		}

		fragment_shader_module = loadShaderModule("./shaders/shader.frag.spv");
		if (!fragment_shader_module) {
			std::cerr << "Failed to load Vulkan fragment shader from file\n";
			veekay::app.running = false;
			return;
		}

		VkPipelineShaderStageCreateInfo stage_infos[2];

		// NOTE: Vertex shader stage
		stage_infos[0] = VkPipelineShaderStageCreateInfo{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
			.stage = VK_SHADER_STAGE_VERTEX_BIT,
			.module = vertex_shader_module,
			.pName = "main",
		};

		// NOTE: Fragment shader stage
		stage_infos[1] = VkPipelineShaderStageCreateInfo{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
			.stage = VK_SHADER_STAGE_FRAGMENT_BIT,
			.module = fragment_shader_module,
			.pName = "main",
		};

		// NOTE: How many bytes does a vertex take?
		VkVertexInputBindingDescription buffer_binding{
			.binding = 0,
			.stride = sizeof(Vertex),
			.inputRate = VK_VERTEX_INPUT_RATE_VERTEX,
		};

		// NOTE: Declare vertex attributes
		VkVertexInputAttributeDescription attributes[] = {
			{
				.location = 0, // NOTE: First attribute
				.binding = 0, // NOTE: First vertex buffer
				.format = VK_FORMAT_R32G32B32_SFLOAT, // NOTE: 3-component vector of floats
				.offset = offsetof(Vertex, position), // NOTE: Offset of "position" field in a Vertex struct
			},
			{
				.location = 1,
				.binding = 0,
				.format = VK_FORMAT_R32G32B32_SFLOAT,
				.offset = offsetof(Vertex, normal),
			},
			{
				.location = 2,
				.binding = 0,
				.format = VK_FORMAT_R32G32_SFLOAT,
				.offset = offsetof(Vertex, uv),
			},
		};

		// NOTE: Describe inputs
		VkPipelineVertexInputStateCreateInfo input_state_info{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
			.vertexBindingDescriptionCount = 1,
			.pVertexBindingDescriptions = &buffer_binding,
			.vertexAttributeDescriptionCount = sizeof(attributes) / sizeof(attributes[0]),
			.pVertexAttributeDescriptions = attributes,
		};

		// NOTE: Every three vertices make up a triangle,
		//       so our vertex buffer contains a "list of triangles"
		VkPipelineInputAssemblyStateCreateInfo assembly_state_info{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
			.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
		};

		// NOTE: Declare clockwise triangle order as front-facing
		//       Discard triangles that are facing away
		//       Fill triangles, don't draw lines instaed
		VkPipelineRasterizationStateCreateInfo raster_info{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
			.polygonMode = VK_POLYGON_MODE_FILL,
			.cullMode = VK_CULL_MODE_BACK_BIT,
			.frontFace = VK_FRONT_FACE_CLOCKWISE,
			.lineWidth = 1.0f,
		};

		// NOTE: Use 1 sample per pixel
		VkPipelineMultisampleStateCreateInfo sample_info{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
			.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT,
			.sampleShadingEnable = false,
			.minSampleShading = 1.0f,
		};

		VkViewport viewport{
			.x = 0.0f,
			.y = 0.0f,
			.width = static_cast<float>(veekay::app.window_width),
			.height = static_cast<float>(veekay::app.window_height),
			.minDepth = 0.0f,
			.maxDepth = 1.0f,
		};

		VkRect2D scissor{
			.offset = {0, 0},
			.extent = {veekay::app.window_width, veekay::app.window_height},
		};

		// NOTE: Let rasterizer draw on the entire window
		VkPipelineViewportStateCreateInfo viewport_info{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,

			.viewportCount = 1,
			.pViewports = &viewport,

			.scissorCount = 1,
			.pScissors = &scissor,
		};

		// NOTE: Let rasterizer perform depth-testing and overwrite depth values on condition pass
		VkPipelineDepthStencilStateCreateInfo depth_info{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
			.depthTestEnable = true,
			.depthWriteEnable = true,
			.depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL,
		};

		// NOTE: Let fragment shader write all the color channels
		VkPipelineColorBlendAttachmentState attachment_info{
			.colorWriteMask = VK_COLOR_COMPONENT_R_BIT |
			                  VK_COLOR_COMPONENT_G_BIT |
			                  VK_COLOR_COMPONENT_B_BIT |
			                  VK_COLOR_COMPONENT_A_BIT,
		};

		// NOTE: Let rasterizer just copy resulting pixels onto a buffer, don't blend
		VkPipelineColorBlendStateCreateInfo blend_info{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,

			.logicOpEnable = false,
			.logicOp = VK_LOGIC_OP_COPY,

			.attachmentCount = 1,
			.pAttachments = &attachment_info
		};

		{
			VkDescriptorPoolSize pools[] = {
				{
					.type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
					.descriptorCount = 8,
				},
				{
					.type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC,
					.descriptorCount = 8,
				},
				{
					.type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
					.descriptorCount = 8,
				},
				{
					.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
					.descriptorCount = 8,
				},

			};
			
			VkDescriptorPoolCreateInfo info{
				.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
				.maxSets = 1,
				.poolSizeCount = sizeof(pools) / sizeof(pools[0]),
				.pPoolSizes = pools,
			};

			if (vkCreateDescriptorPool(device, &info, nullptr,
			                           &descriptor_pool) != VK_SUCCESS) {
				std::cerr << "Failed to create Vulkan descriptor pool\n";
				veekay::app.running = false;
				return;
			}
		}

		// NOTE: Descriptor set layout specification
		{
			VkDescriptorSetLayoutBinding bindings[] = {
				{
					.binding = 0,
					.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
					.descriptorCount = 1,
					.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
				},
				{
					.binding = 1,
					.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC,
					.descriptorCount = 1,
					.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
				},
				{
					.binding = 2,
					.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
					.descriptorCount = 1,
					.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
				},
				{
					.binding = 3,
					.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
					.descriptorCount = 1,
					.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
				},
			};

			VkDescriptorSetLayoutCreateInfo info{
				.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
				.bindingCount = sizeof(bindings) / sizeof(bindings[0]),
				.pBindings = bindings,
			};

			if (vkCreateDescriptorSetLayout(device, &info, nullptr,
			                                &descriptor_set_layout) != VK_SUCCESS) {
				std::cerr << "Failed to create Vulkan descriptor set layout\n";
				veekay::app.running = false;
				return;
			}
		}

		{
			VkDescriptorSetAllocateInfo info{
				.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
				.descriptorPool = descriptor_pool,
				.descriptorSetCount = 1,
				.pSetLayouts = &descriptor_set_layout,
			};

			if (vkAllocateDescriptorSets(device, &info, &descriptor_set) != VK_SUCCESS) {
				std::cerr << "Failed to create Vulkan descriptor set\n";
				veekay::app.running = false;
				return;
			}
		}

		// NOTE: Declare external data sources, only push constants this time
		VkPipelineLayoutCreateInfo layout_info{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
			.setLayoutCount = 1,
			.pSetLayouts = &descriptor_set_layout,
		};

		// NOTE: Create pipeline layout
		if (vkCreatePipelineLayout(device, &layout_info,
		                           nullptr, &pipeline_layout) != VK_SUCCESS) {
			std::cerr << "Failed to create Vulkan pipeline layout\n";
			veekay::app.running = false;
			return;
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
			.pDepthStencilState = &depth_info,
			.pColorBlendState = &blend_info,
			.layout = pipeline_layout,
			.renderPass = veekay::app.vk_render_pass,
		};

		// NOTE: Create graphics pipeline
		if (vkCreateGraphicsPipelines(device, nullptr,
		                              1, &info, nullptr, &pipeline) != VK_SUCCESS) {
			std::cerr << "Failed to create Vulkan pipeline\n";
			veekay::app.running = false;
			return;
		}
	}

	scene_uniforms_buffer = new veekay::graphics::Buffer(
		sizeof(SceneUniforms),
		nullptr,
		VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);

	model_uniforms_buffer = new veekay::graphics::Buffer(
		max_models * veekay::graphics::Buffer::structureAlignment(sizeof(ModelUniforms)),
		nullptr,
		VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);

	point_lights_buffer = new veekay::graphics::Buffer(
		max_point_lights * sizeof(PointLight),
		nullptr,
		VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
	);

	spot_lights_buffer = new veekay::graphics::Buffer(
		max_spot_lights * sizeof(SpotLight),
		nullptr,
		VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
	);

	// NOTE: This texture and sampler is used when texture could not be loaded
	{
		VkSamplerCreateInfo info{
			.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
			.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
		};

		if (vkCreateSampler(device, &info, nullptr, &missing_texture_sampler) != VK_SUCCESS) {
			std::cerr << "Failed to create Vulkan texture sampler\n";
			veekay::app.running = false;
			return;
		}

		uint32_t pixels[] = {
			0xff000000, 0xffff00ff,
			0xffff00ff, 0xff000000,
		};

		missing_texture = new veekay::graphics::Texture(cmd, 2, 2,
		                                                VK_FORMAT_B8G8R8A8_UNORM,
		                                                pixels);
	}

	{
		VkDescriptorBufferInfo buffer_infos[] = {
			{
				.buffer = scene_uniforms_buffer->buffer,
				.offset = 0,
				.range = sizeof(SceneUniforms),
			},
			{
				.buffer = model_uniforms_buffer->buffer,
				.offset = 0,
				.range = sizeof(ModelUniforms),
			},
			{
				.buffer = point_lights_buffer->buffer,
				.offset = 0,
				.range = max_point_lights * sizeof(PointLight),
			},
			{
				.buffer = spot_lights_buffer->buffer,
				.offset = 0,
				.range = max_spot_lights * sizeof(SpotLight),
			},
		};

		VkWriteDescriptorSet write_infos[] = {
			{
				.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
				.dstSet = descriptor_set,
				.dstBinding = 0,
				.dstArrayElement = 0,
				.descriptorCount = 1,
				.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
				.pBufferInfo = &buffer_infos[0],
			},
			{
				.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
				.dstSet = descriptor_set,
				.dstBinding = 1,
				.dstArrayElement = 0,
				.descriptorCount = 1,
				.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC,
				.pBufferInfo = &buffer_infos[1],
			},
			{
				.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
				.dstSet = descriptor_set,
				.dstBinding = 2,
				.dstArrayElement = 0,
				.descriptorCount = 1,
				.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
				.pBufferInfo = &buffer_infos[2],
			},
			{
				.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
				.dstSet = descriptor_set,
				.dstBinding = 3,
				.dstArrayElement = 0,
				.descriptorCount = 1,
				.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
				.pBufferInfo = &buffer_infos[3],
			},
		};

		vkUpdateDescriptorSets(device, sizeof(write_infos) / sizeof(write_infos[0]),
		                       write_infos, 0, nullptr);
	}

	// NOTE: Plane mesh initialization
	{
		// (v0)------(v1)
		//  |  \       |
		//  |   `--,   |
		//  |       \  |
		// (v3)------(v2)
		std::vector<Vertex> vertices = {
			{{-5.0f, 0.0f, 5.0f}, {0.0f, -1.0f, 0.0f}, {0.0f, 0.0f}},
			{{5.0f, 0.0f, 5.0f}, {0.0f, -1.0f, 0.0f}, {1.0f, 0.0f}},
			{{5.0f, 0.0f, -5.0f}, {0.0f, -1.0f, 0.0f}, {1.0f, 1.0f}},
			{{-5.0f, 0.0f, -5.0f}, {0.0f, -1.0f, 0.0f}, {0.0f, 1.0f}},
		};

		std::vector<uint32_t> indices = {
			0, 1, 2, 2, 3, 0
		};

		plane_mesh.vertex_buffer = new veekay::graphics::Buffer(
			vertices.size() * sizeof(Vertex), vertices.data(),
			VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);

		plane_mesh.index_buffer = new veekay::graphics::Buffer(
			indices.size() * sizeof(uint32_t), indices.data(),
			VK_BUFFER_USAGE_INDEX_BUFFER_BIT);

		plane_mesh.indices = uint32_t(indices.size());
	}

	// NOTE: Cube mesh initialization
	{
		std::vector<Vertex> vertices = {
			{{-0.5f, -0.5f, -0.5f}, {0.0f, 0.0f, -1.0f}, {0.0f, 0.0f}},
			{{+0.5f, -0.5f, -0.5f}, {0.0f, 0.0f, -1.0f}, {1.0f, 0.0f}},
			{{+0.5f, +0.5f, -0.5f}, {0.0f, 0.0f, -1.0f}, {1.0f, 1.0f}},
			{{-0.5f, +0.5f, -0.5f}, {0.0f, 0.0f, -1.0f}, {0.0f, 1.0f}},

			{{+0.5f, -0.5f, -0.5f}, {1.0f, 0.0f, 0.0f}, {0.0f, 0.0f}},
			{{+0.5f, -0.5f, +0.5f}, {1.0f, 0.0f, 0.0f}, {1.0f, 0.0f}},
			{{+0.5f, +0.5f, +0.5f}, {1.0f, 0.0f, 0.0f}, {1.0f, 1.0f}},
			{{+0.5f, +0.5f, -0.5f}, {1.0f, 0.0f, 0.0f}, {0.0f, 1.0f}},

			{{+0.5f, -0.5f, +0.5f}, {0.0f, 0.0f, 1.0f}, {0.0f, 0.0f}},
			{{-0.5f, -0.5f, +0.5f}, {0.0f, 0.0f, 1.0f}, {1.0f, 0.0f}},
			{{-0.5f, +0.5f, +0.5f}, {0.0f, 0.0f, 1.0f}, {1.0f, 1.0f}},
			{{+0.5f, +0.5f, +0.5f}, {0.0f, 0.0f, 1.0f}, {0.0f, 1.0f}},

			{{-0.5f, -0.5f, +0.5f}, {-1.0f, 0.0f, 0.0f}, {0.0f, 0.0f}},
			{{-0.5f, -0.5f, -0.5f}, {-1.0f, 0.0f, 0.0f}, {1.0f, 0.0f}},
			{{-0.5f, +0.5f, -0.5f}, {-1.0f, 0.0f, 0.0f}, {1.0f, 1.0f}},
			{{-0.5f, +0.5f, +0.5f}, {-1.0f, 0.0f, 0.0f}, {0.0f, 1.0f}},

			{{-0.5f, -0.5f, +0.5f}, {0.0f, -1.0f, 0.0f}, {0.0f, 0.0f}},
			{{+0.5f, -0.5f, +0.5f}, {0.0f, -1.0f, 0.0f}, {1.0f, 0.0f}},
			{{+0.5f, -0.5f, -0.5f}, {0.0f, -1.0f, 0.0f}, {1.0f, 1.0f}},
			{{-0.5f, -0.5f, -0.5f}, {0.0f, -1.0f, 0.0f}, {0.0f, 1.0f}},

			{{-0.5f, +0.5f, -0.5f}, {0.0f, 1.0f, 0.0f}, {0.0f, 0.0f}},
			{{+0.5f, +0.5f, -0.5f}, {0.0f, 1.0f, 0.0f}, {1.0f, 0.0f}},
			{{+0.5f, +0.5f, +0.5f}, {0.0f, 1.0f, 0.0f}, {1.0f, 1.0f}},
			{{-0.5f, +0.5f, +0.5f}, {0.0f, 1.0f, 0.0f}, {0.0f, 1.0f}},
		};

		std::vector<uint32_t> indices = {
			0, 1, 2, 2, 3, 0,
			4, 5, 6, 6, 7, 4,
			8, 9, 10, 10, 11, 8,
			12, 13, 14, 14, 15, 12,
			16, 17, 18, 18, 19, 16,
			20, 21, 22, 22, 23, 20,
		};

		cube_mesh.vertex_buffer = new veekay::graphics::Buffer(
			vertices.size() * sizeof(Vertex), vertices.data(),
			VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);

		cube_mesh.index_buffer = new veekay::graphics::Buffer(
			indices.size() * sizeof(uint32_t), indices.data(),
			VK_BUFFER_USAGE_INDEX_BUFFER_BIT);

		cube_mesh.indices = uint32_t(indices.size());
	}

	// NOTE: Add models to scene
	models.emplace_back(Model{
		.mesh = plane_mesh,
		.transform = Transform{},
		.albedo_color = veekay::vec3{1.0f, 1.0f, 1.0f}
	});

	models.emplace_back(Model{
		.mesh          = cube_mesh,
		.transform     = Transform{
			{-2.0f, -0.5f, -1.5f},
		},
		.albedo_color  = { 1.0f, 0.1f, 0.1f },
		.specular_color= { 0.05f, 0.05f, 0.05f }, // почти нет блика
		.shininess     = 8.0f,                   // широкий, размазанный блик
	});

	// Центральный куб — глянцевый синий пластик
	models.emplace_back(Model{
		.mesh          = cube_mesh,
		.transform     = Transform{
			.position = {1.5f, -0.5f, -0.5f},
		},
		.albedo_color  = { 0.1f, 0.1f, 1.0f },
		.specular_color= { 0.6f, 0.6f, 0.6f },   // заметный блик
		.shininess     = 32.0f,                  // более узкий блик
	});

	// Правый куб — “металл” / хром
	models.emplace_back(Model{
		.mesh          = cube_mesh,
		.transform     = Transform{
			.position = {0.0f, -0.5f, 1.0f},
		},
		.albedo_color  = { 0.1f, 0.9f, 0.1f },
		.specular_color= { 1.0f, 1.0f, 1.0f },   // очень сильный белый блик
		.shininess     = 128.0f,                 // острый маленький блик
	});

	point_lights.emplace_back(PointLight{
		.position = { 0.0f, 3.0f, 3.0f },   // над центром сцены, чуть ближе к камере
		.radius   = 12.0f,
		.color    = { 1.0f, 0.95f, 0.9f },  // тёплый белый
	});

	spot_lights.emplace_back(SpotLight{
		.position  = { -4.0f, 3.0f, 0.0f },                      // слева сверху
		.radius    = 15.0f,
		.direction = veekay::vec3::normalized({ 1.0f, -0.6f, 0.2f }),
		.angle = std::cosf(toRadians(20.0f)),               // узкий конус ~20°
		.color     = { 0.8f, 0.9f, 1.0f },                      // чуть голубоватый
	});
}

// NOTE: Destroy resources here, do not cause leaks in your program!
void shutdown() {
	VkDevice& device = veekay::app.vk_device;

	vkDestroySampler(device, missing_texture_sampler, nullptr);
	delete missing_texture;

	delete cube_mesh.index_buffer;
	delete cube_mesh.vertex_buffer;

	delete plane_mesh.index_buffer;
	delete plane_mesh.vertex_buffer;

	delete model_uniforms_buffer;
	delete scene_uniforms_buffer;

	delete point_lights_buffer;
	delete spot_lights_buffer;

	vkDestroyDescriptorSetLayout(device, descriptor_set_layout, nullptr);
	vkDestroyDescriptorPool(device, descriptor_pool, nullptr);

	vkDestroyPipeline(device, pipeline, nullptr);
	vkDestroyPipelineLayout(device, pipeline_layout, nullptr);
	vkDestroyShaderModule(device, fragment_shader_module, nullptr);
	vkDestroyShaderModule(device, vertex_shader_module, nullptr);
}

void update(double time) {
	ImGui::Begin("Controls:");
	ImGui::Checkbox("Use LookAt", &g_use_lookat);
	ImGui::SliderFloat("Mouse sensitivity",
                   &g_mouse_sensitivity,
                   0.001f, 0.01f,
                   "%.4f rad/pix");
	ImGui::Text("LMB: look | WASD: move XZ | Q/Z: up/down");
	ImGui::Separator();

    // --- Солнце (направленный свет) ---
    if (ImGui::CollapsingHeader("Sun light", ImGuiTreeNodeFlags_DefaultOpen)) {
        // Цвет солнца
        ImGui::ColorEdit3("Sun color", g_sun_color.elements);

        // Направление солнца
        veekay::vec3 sun_dir_tmp = g_sun_dir;
        if (ImGui::DragFloat3("Sun direction", sun_dir_tmp.elements, 0.01f, -1.0f, 1.0f)) {
            // Нормализуем, чтобы не словить странности в шейдере
            g_sun_dir = veekay::vec3::normalized(sun_dir_tmp);
        }

        // Ambient
        ImGui::ColorEdit3("Ambient", g_ambient.elements);
    }

    ImGui::Separator();

    // --- Первый точечный источник ---
    if (!point_lights.empty() &&
        ImGui::CollapsingHeader("Point light 0", ImGuiTreeNodeFlags_DefaultOpen)) {

        PointLight& pl = point_lights[0];

        ImGui::DragFloat3("Position##PL0", pl.position.elements, 0.05f, -10.0f, 10.0f);
        ImGui::SliderFloat("Radius##PL0", &pl.radius, 0.1f, 20.0f);
        ImGui::ColorEdit3("Color##PL0", pl.color.elements);
    }

    ImGui::Separator();

    // --- Первый прожектор ---
    if (!spot_lights.empty() &&
        ImGui::CollapsingHeader("Spot light 0", ImGuiTreeNodeFlags_DefaultOpen)) {

        SpotLight& sl = spot_lights[0];

        ImGui::DragFloat3("Position##SL0", sl.position.elements, 0.05f, -10.0f, 10.0f);
        ImGui::SliderFloat("Radius##SL0", &sl.radius, 0.1f, 20.0f);

        // Направление — редактируем, потом нормализуем
        veekay::vec3 dir_tmp = sl.direction;
        if (ImGui::DragFloat3("Direction##SL0", dir_tmp.elements, 0.01f, -1.0f, 1.0f)) {
            sl.direction = veekay::vec3::normalized(dir_tmp);
        }

        // Угол конуса в градусах (в структуре хранится cos(angle))
        float angle_deg = std::acosf(std::clamp(sl.angle, -1.0f, 1.0f)) * 180.0f / float(M_PI);
        if (ImGui::SliderFloat("Angle (deg)##SL0", &angle_deg, 5.0f, 60.0f)) {
            sl.angle = std::cosf(toRadians(angle_deg));
        }

        ImGui::ColorEdit3("Color##SL0", sl.color.elements);
    }

    ImGui::End();

	if (!ImGui::IsWindowHovered()) {
		using namespace veekay::input;

		// Поворот мышью при зажатой ЛКМ
		if (mouse::isButtonDown(mouse::Button::left)) {
			auto delta = mouse::cursorDelta();

			// rotation.x, rotation.y — в радианах
			camera.rotation.y += delta.x * g_mouse_sensitivity;
			camera.rotation.x += delta.y * g_mouse_sensitivity;

			// Кламп pitch, нормализация yaw (в радианах)
			const float pitch_limit = toRadians(89.0f);
			const float yaw_limit   = float(M_PI); // π ~ 180°

			if (camera.rotation.x >  pitch_limit) camera.rotation.x =  pitch_limit;
			if (camera.rotation.x < -pitch_limit) camera.rotation.x = -pitch_limit;

			if (camera.rotation.y >  yaw_limit)   camera.rotation.y -= 2.0f * yaw_limit; // 2π
			if (camera.rotation.y < -yaw_limit)   camera.rotation.y += 2.0f * yaw_limit;
		}

		// Обновляем локальный базис из yaw/pitch
		veekay::vec3 right, up, front;
		yawPitchBasis(camera.rotation.y, camera.rotation.x, right, up, front);

		// Движение в плоскости XZ относительно направления взгляда
		veekay::vec3 front_xz = veekay::vec3::normalized({front.x, 0.0f, front.z});
		veekay::vec3 right_xz = veekay::vec3::normalized({right.x, 0.0f, right.z});
		const float move = 0.1f;

		if (keyboard::isKeyDown(keyboard::Key::w)) camera.position -= front_xz * move;
		if (keyboard::isKeyDown(keyboard::Key::s)) camera.position += front_xz * move;
		if (keyboard::isKeyDown(keyboard::Key::d)) camera.position -= right_xz * move;
		if (keyboard::isKeyDown(keyboard::Key::a)) camera.position += right_xz * move;

		// Вертикаль: Q вверх, Z вниз (WORLD_UP = {0,-1,0})
		if (keyboard::isKeyDown(keyboard::Key::q)) camera.position.y -= move;
		if (keyboard::isKeyDown(keyboard::Key::z)) camera.position.y += move;
	}

	float aspect_ratio = float(veekay::app.window_width) / float(veekay::app.window_height);
	SceneUniforms scene_uniforms{
		.view_projection = camera.view_projection(aspect_ratio),
		.view_position = camera.position,
		.ambient_light_intensity = g_ambient,
		.sun_light_direction = g_sun_dir,
		.sun_light_color = g_sun_color,
		.point_lights_count = static_cast<uint32_t>(point_lights.size()),
		.spot_lights_count = static_cast<uint32_t>(spot_lights.size())
	};

	std::vector<ModelUniforms> model_uniforms(models.size());
	for (size_t i = 0, n = models.size(); i < n; ++i) {
		const Model& model = models[i];
		ModelUniforms& uniforms = model_uniforms[i];

		uniforms.model = model.transform.matrix();
		uniforms.albedo_color = model.albedo_color;
		uniforms.model         = model.transform.matrix();
		uniforms.albedo_color  = model.albedo_color;
		uniforms.specular_color = model.specular_color;
		uniforms.shininess     = model.shininess;
	}

	*(SceneUniforms*)scene_uniforms_buffer->mapped_region = scene_uniforms;

	const size_t model_stride = veekay::graphics::Buffer::structureAlignment(sizeof(ModelUniforms));
	const size_t spot_stride  = sizeof(SpotLight);
	const size_t point_stride = sizeof(PointLight);

	for (size_t i = 0, n = model_uniforms.size(); i < n; ++i) {
		const ModelUniforms& uniforms = model_uniforms[i];

		char* const pointer = static_cast<char*>(model_uniforms_buffer->mapped_region) + i * model_stride;
		*reinterpret_cast<ModelUniforms*>(pointer) = uniforms;
	}

	if (point_lights_buffer) {
        if (!point_lights.empty()) {
            std::memcpy(point_lights_buffer->mapped_region, 
                       point_lights.data(), 
                       point_lights.size() * sizeof(PointLight));
		}
    }
    
    if (spot_lights_buffer) {
        if (!spot_lights.empty()) {
            std::memcpy(spot_lights_buffer->mapped_region, 
                       spot_lights.data(), 
                       spot_lights.size() * sizeof(SpotLight));
        } 
    }
}

void render(VkCommandBuffer cmd, VkFramebuffer framebuffer) {
	vkResetCommandBuffer(cmd, 0);

	{ // NOTE: Start recording rendering commands
		VkCommandBufferBeginInfo info{
			.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
			.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
		};

		vkBeginCommandBuffer(cmd, &info);
	}

	{ // NOTE: Use current swapchain framebuffer and clear it
		VkClearValue clear_color{.color = {{0.1f, 0.1f, 0.1f, 1.0f}}};
		VkClearValue clear_depth{.depthStencil = {1.0f, 0}};

		VkClearValue clear_values[] = {clear_color, clear_depth};

		VkRenderPassBeginInfo info{
			.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
			.renderPass = veekay::app.vk_render_pass,
			.framebuffer = framebuffer,
			.renderArea = {
				.extent = {
					veekay::app.window_width,
					veekay::app.window_height
				},
			},
			.clearValueCount = 2,
			.pClearValues = clear_values,
		};

		vkCmdBeginRenderPass(cmd, &info, VK_SUBPASS_CONTENTS_INLINE);
	}

	vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
	VkDeviceSize zero_offset = 0;

	VkBuffer current_vertex_buffer = VK_NULL_HANDLE;
	VkBuffer current_index_buffer = VK_NULL_HANDLE;

	const size_t model_uniorms_alignment =
		veekay::graphics::Buffer::structureAlignment(sizeof(ModelUniforms));

	for (size_t i = 0, n = models.size(); i < n; ++i) {
		const Model& model = models[i];
		const Mesh& mesh = model.mesh;

		if (current_vertex_buffer != mesh.vertex_buffer->buffer) {
			current_vertex_buffer = mesh.vertex_buffer->buffer;
			vkCmdBindVertexBuffers(cmd, 0, 1, &current_vertex_buffer, &zero_offset);
		}

		if (current_index_buffer != mesh.index_buffer->buffer) {
			current_index_buffer = mesh.index_buffer->buffer;
			vkCmdBindIndexBuffer(cmd, current_index_buffer, zero_offset, VK_INDEX_TYPE_UINT32);
		}

		uint32_t offset = i * model_uniorms_alignment;
		vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_layout,
		                    0, 1, &descriptor_set, 1, &offset);

		vkCmdDrawIndexed(cmd, mesh.indices, 1, 0, 0, 0);
	}

	vkCmdEndRenderPass(cmd);
	vkEndCommandBuffer(cmd);
}

} // namespace

int main() {
	return veekay::run({
		.init = initialize,
		.shutdown = shutdown,
		.update = update,
		.render = render,
	});
}
