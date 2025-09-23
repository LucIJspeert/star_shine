"""STAR SHINE
Satellite Time-series Analysis Routine using Sinusoids and Harmonics through Iterative Non-linear Extraction

This module contains (data) descriptors.
"""


class ValidType:
    """Data descriptor with type check and optional default or default factory.

    This descriptor validates that assigned values match the specified type.
    It also supports providing a default value, which can be either:
    - a static value (e.g., int, str, tuple), or
    - a callable factory that produces a fresh value per instance (useful for mutable defaults like list or dict).

    Attributes
    ----------
    property_name: str
        The name of the attribute as defined in the owning class (set automatically by __set_name__).
    """
    def __init__(self, valid_type, default=None):
        """Initialize a ValidType descriptor.

        Parameters
        ----------
        valid_type: type
            The expected type for the attribute. Assigning a value of a different type will raise a TypeError.
        default: any or callable, optional
            A default value to use if the attribute is accessed before being explicitly set.
            If a callable is provided, it will be called to generate a fresh value per instance
            If None, no default is provided, and accessing the attribute before setting it will return None.

        Raises
        ------
        TypeError
            If a non-callable default is provided that does not match the specified type.
        """
        self._valid_type = valid_type
        self._default = default
        self.property_name = None

        # If default is static (not callable), validate it immediately
        if default is not None and not callable(default) and not isinstance(default, valid_type):
            raise TypeError(
                f"Default must be of type {valid_type.__name__}, "
                f"not {type(default).__name__}"
            )

    def __set_name__(self, owner_class, property_name):
        """Called automatically when the descriptor is assigned to a class attribute.

        Parameters
        ----------
        owner_class: type
            The class owning this descriptor.
        property_name: str
            The name of the attribute in the owner class.
        """
        self.property_name = property_name

    def __set__(self, instance, value):
        """Assign a value to the attribute, enforcing type checking.

        Parameters
        ----------
        instance: object
            The instance on which the attribute is being set.
        value: any
            The value to assign to the attribute.

        Raises
        ------
        TypeError
            If the value is not of the expected type.
        """
        if not isinstance(value, self._valid_type):
            raise TypeError(
                f'{self.property_name} must be of type {self._valid_type.__name__}, '
                f'not {type(value).__name__}'
            )

        instance.__dict__[self.property_name] = value

        return None

    def __get__(self, instance, owner_class):
        """Retrieve the value of the attribute. If not set, return or generate the default.

        Parameters
        ----------
        instance: object
            The instance on which the attribute is being accessed.
        owner_class: type
            The class owning this descriptor.

        Returns
        -------
        any
            The current value of the attribute, or the default value if it has not been set.

        Raises
        ------
        TypeError
            If a callable default is provided that does not generate the specified type.
        """
        if instance is None:
            return self

        # check for presence in dict, set default value if given
        if self.property_name not in instance.__dict__:
            if callable(self._default):
                # call the factory
                generated_value = self._default()

                if not isinstance(generated_value, self._valid_type):
                    raise TypeError(
                        f"Default factory for {self.property_name} must produce a "
                        f"{self._valid_type.__name__}, not {type(generated_value).__name__}"
                    )

                instance.__dict__[self.property_name] = generated_value
            else:
                # static default
                instance.__dict__[self.property_name] = self._default

        return instance.__dict__[self.property_name]
