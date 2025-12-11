#version 450

layout (location = 0) in vec3 v_position;
layout (location = 1) in vec3 v_normal;
layout (location = 2) in vec2 v_uv;

layout (location = 0) out vec3 f_position;
layout (location = 1) out vec3 f_normal;
layout (location = 2) out vec2 f_uv;
layout (location = 3) out vec4 f_shadow_position;

layout (binding = 0, std140) uniform SceneUniforms {
    mat4 view_projection;
    mat4 shadow_projection;
    vec3 view_position;  float _pad0;
    vec3 ambient_light_intensity; float _pad1;
    vec3 sun_light_direction; float _pad2;
    vec3 sun_light_color; float _pad3;
    uint point_lights_count;
    uint spot_lights_count;
    uvec2 _pad4;
    float curve; float _pad8[3];
};

layout (binding = 1, std140) uniform ModelUniforms {
    mat4 model;
    vec3 albedo_color;   float _pad5;
    vec3 specular_color; float _pad6;
    float shininess;     vec3 _pad7;
};

void main() {
	vec4 position = model * vec4(v_position, 1.0f);
	vec4 normal = model * vec4(v_normal, 0.0f);

	gl_Position = view_projection * position;

    f_shadow_position = shadow_projection * position;
	f_position = position.xyz;
	f_normal = normal.xyz;
	f_uv = v_uv;
}
