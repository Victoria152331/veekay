#version 450

struct PointLight {
    vec3 position;
    float radius;
    vec3 color;
    float _pad00;
};

struct SpotLight {
    vec3 position;
    float radius;
    vec3 direction;
    float angle_cos;
    vec3 color;
    float _pad11;
};

layout (location = 0) in vec3 f_position;
layout (location = 1) in vec3 f_normal;
layout (location = 2) in vec2 f_uv;

layout (location = 0) out vec4 final_color;

layout(binding = 0, std140) uniform SceneUniforms {
    mat4 view_projection;
    vec3 view_position; float _pad0;
    vec3 ambient_light_intensity; float _pad1;
    vec3 sun_light_direction; float _pad2;
    vec3 sun_light_color; float _pad3;
    uint point_lights_count;
    uint spot_lights_count;
    uvec2 _pad4;
    float curve; float _pad5[3];
} scene;

layout (binding = 1, std140) uniform ModelUniforms {
    mat4 model;
    vec3 albedo_color; float _pad10;
    vec3 specular_color; float _pad12;
    float shininess;
    uint _pad13[3];
} model;

layout(binding = 2, std430) readonly buffer PointLights {
    PointLight point_lights[];
};

layout(binding = 3, std430) readonly buffer SpotLights {
    SpotLight spot_lights[];
};

layout (binding = 4) uniform sampler2D albedo_texture;

layout (binding = 5) uniform sampler2D specular_texture;

layout (binding = 6) uniform sampler2D emissive_texture;


vec3 calculateBlinnPhong(vec3 lightDir, vec3 normal, vec3 viewDir, vec3 lightColor, vec4 albedo_texel, vec4 spec_texel) {
    float diff = max(dot(normal, lightDir), 0.0);
    vec3 diffuse = diff * albedo_texel.xyz * model.albedo_color * lightColor;
    
    vec3 halfwayDir = normalize(lightDir + viewDir);
    float spec = pow(max(dot(normal, halfwayDir), 0.0), model.shininess);
    vec3 specular = spec * model.specular_color * spec_texel.xyz * lightColor;
    
    return diffuse + specular;
}

void main() {
    vec3 normal = normalize(f_normal);
    vec3 viewDir = normalize(scene.view_position - f_position);

    vec2 strange_uv = { f_uv.x + cos(f_uv.y * 100) / 100, f_uv.y};
    vec2 strange_curve_uv = { f_uv.x + cos(f_uv.y * 100)  * scene.curve / 100, f_uv.y};

    vec2 cur_uv = strange_curve_uv;
    vec4 albedo_texel = texture(albedo_texture, cur_uv);
    vec4 spec_texel = texture(specular_texture, cur_uv);
    vec4 emissive_texel = texture(emissive_texture, cur_uv);
    
    vec3 result = albedo_texel.xyz * model.albedo_color * scene.ambient_light_intensity;
    
    vec3 sunLightDir = normalize(-scene.sun_light_direction);
    result += calculateBlinnPhong(sunLightDir, normal, viewDir, scene.sun_light_color, albedo_texel, spec_texel);
    
    for (uint i = 0u; i < scene.point_lights_count; i++) {
        PointLight light = point_lights[i];
        vec3 lightDir = normalize(light.position - f_position);
        float distance = length(light.position - f_position);
        
        if (distance > light.radius) continue;

		float ndotl = dot(normal, lightDir);
    	if (ndotl <= 0.0) continue;
        
        float attenuation = 1.0 - (distance / max(light.radius, 0.0001));
        attenuation *= attenuation;
        
        vec3 lighting = calculateBlinnPhong(lightDir, normal, viewDir, light.color, albedo_texel, spec_texel);
        result += lighting * attenuation;
    }
    
    for (uint i = 0u; i < scene.spot_lights_count; i++) {
        SpotLight light = spot_lights[i];
        vec3 lightDir = normalize(light.position - f_position);
        float distance = length(light.position - f_position);
        
        if (distance > light.radius) continue;

		float ndotl = dot(normal, lightDir);
    	if (ndotl <= 0.0) continue;
        
        float cosTheta = dot(lightDir, normalize(-light.direction));
        if (cosTheta < light.angle_cos) continue;
        
        float distanceAttenuation = 1.0 - (distance / max(light.radius, 0.0001));
        distanceAttenuation *= distanceAttenuation;

        float softEdge = 0.1;
        float angleAttenuation = clamp((cosTheta - light.angle_cos) / softEdge, 0.0, 1.0);
        
        vec3 lighting = calculateBlinnPhong(lightDir, normal, viewDir, light.color, albedo_texel, spec_texel);
        result += lighting * distanceAttenuation * angleAttenuation;
    }
    result += emissive_texel.xyz;
    if (result.x > 1.0) result.x = 1.0;
    if (result.y > 1.0) result.y = 1.0;
    if (result.z > 1.0) result.z = 1.0;
    final_color = vec4(result, 1.0);
}
