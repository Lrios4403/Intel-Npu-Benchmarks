# Read-only dump of the devices OpenVINO sees and their readable properties.
import openvino as ov

core = ov.Core()
print("openvino", ov.get_version())
print("available devices:", core.available_devices)

SKIP = {"SUPPORTED_PROPERTIES", "SUPPORTED_METRICS", "SUPPORTED_CONFIG_KEYS", "CACHING_PROPERTIES"}
for device in core.available_devices:
    print(f"\n=== {device}")
    for prop in core.get_property(device, "SUPPORTED_PROPERTIES"):
        if prop in SKIP:
            continue
        try:
            value = core.get_property(device, prop)
        except Exception as error:  # some properties are write-only or need a compiled model
            continue
        text = str(value)
        if len(text) > 160:
            text = text[:160] + "..."
        print(f"  {prop}: {text}")
