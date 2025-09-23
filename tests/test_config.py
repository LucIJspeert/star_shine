import unittest
from star_shine.config.config import Config, get_config, get_config_path
from star_shine.config.descriptors import ValidType

class TestConfig(unittest.TestCase):
    def test_validate_config_valid_data(self):
        """Test with valid config data taken directly from the class"""
        valid_config = {key: getattr(Config, key) for key in dir(Config)
                        if not callable(getattr(Config, key)) and not key.startswith('_')}
        try:
            Config._validate_config(valid_config)
        except ValueError as e:
            self.fail(f"Validation failed for valid configuration: {e}")

    def test_validate_config_missing_keys(self):
        """Test with config data that is missing keys"""
        valid_config = {key: getattr(Config, key) for key in dir(Config)
                        if not callable(getattr(Config, key)) and not key.startswith('_')}

        # copy and remove two items
        incomplete_config = valid_config.copy()
        incomplete_config.popitem()
        incomplete_config.popitem()

        with self.assertRaises(ValueError) as context:
            Config._validate_config(incomplete_config)

        self.assertIn("Missing keys in configuration file", str(context.exception))

    def test_validate_config_excess_keys(self):
        """Test with config data that has excess keys"""
        valid_config = {key: getattr(Config, key) for key in dir(Config)
                        if not callable(getattr(Config, key)) and not key.startswith('_')}

        # copy and add two items
        excess_config = valid_config.copy()
        excess_config['extra_key1'] = 'extra_value1'
        excess_config['extra_key2'] = 'extra_value2'

        with self.assertRaises(ValueError) as context:
            Config._validate_config(excess_config)

        self.assertIn("Excess keys in configuration file", str(context.exception))

    def test_config_loads(self):
        """Check if the configuration loads without errors"""
        try:
            Config()
        except Exception as e:
            self.fail(f"Configuration failed to load: {e}")

    def test_config_getter(self):
        """Check if the configuration loads without errors"""
        try:
            config = get_config()
        except Exception as e:
            self.fail(f"Configuration failed to be gotten: {e}")

        self.assertIsInstance(config, Config)

    def test_config_updates_from_file(self):
        """Check if the configuration updates from file without errors"""
        try:
            config = Config()
            config.update_from_file(get_config_path())
        except Exception as e:
            self.fail(f"Configuration failed to update from file: {e}")

    def test_config_updates_from_dict(self):
        """Check if the configuration updates from dictionary without errors"""
        # build the configuration settings dictionary with the defaults
        valid_config = {}
        for key in dir(Config):
            # avoid methods and private attributes
            if not callable(getattr(Config, key)) and not key.startswith('_'):
                # the use of descriptors requires some extra logic here
                attr = getattr(Config, key)
                if isinstance(attr, ValidType):
                    default_value = attr._default() if callable(attr._default) else attr._default
                else:
                    default_value = attr
                valid_config[key] = default_value

        # copy and remove two items
        incomplete_but_valid_config = valid_config.copy()
        incomplete_but_valid_config.popitem()
        incomplete_but_valid_config.popitem()

        try:
            config = Config()
            config.update_from_dict(incomplete_but_valid_config)
        except Exception as e:
            self.fail(f"Configuration failed to update from dictionary: {e}")

if __name__ == '__main__':
    unittest.main()