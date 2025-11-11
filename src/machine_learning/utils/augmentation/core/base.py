from typing import Union, Any, Callable

import random
from abc import ABC, abstractmethod


class TransformBase(ABC):
    """
    Base class for all transforms in augmentation.

    Args:
        p (float): The probability of applying this transform.
    """

    _targets: tuple[str]  # targets that this transform can work on
    _annotation_targets: tuple[str]  # annotation targets that need to be update with targets

    def __init__(self, p: float = 1):
        self._p = p
        self._params = {}
        self._available_keys = set()
        self._key2func = {}
        self.set_keys()

    @property
    def p(self) -> float:
        return self._p

    @property
    def params(self) -> dict[str, Any]:
        return self._params

    @property
    def available_keys(self) -> set[str]:
        return self._available_keys

    @property
    def key2func(self) -> dict[str, Callable]:
        return self._key2func

    @property
    @abstractmethod
    def targets(self) -> dict[str, Callable]:
        """Get mapping of target keys to their corresponding processing functions.

        Returns:
            dict[str, Callable]: Dictionary mapping target keys to their processing functions.

        """
        # mapping for targets and methods for which they depend
        # for example:
        # >>  {"image": self.apply_to_image}
        # >>  {"masks": self.apply_to_masks}
        raise NotImplementedError

    def set_keys(self) -> None:
        """Set _available_keys."""
        if hasattr(self, "_targets") and len(self._targets) > 0:
            self._available_keys.update(self._targets)
        if hasattr(self, "_annotation_targets") and len(self._annotation_targets) > 0:
            self._available_keys.update(self._annotation_targets)

        self._available_keys.update(self.targets.keys())
        self._key2func = {key: self.targets[key] for key in self._available_keys if key in self.targets}

    def get_params(self) -> dict[str, Any]:
        """
        Obtain transformation parameters independent of input.
        """
        return {}

    def get_params_on_sample(self, sample: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
        """Returns parameters dependent on input sample."""
        return {}

    @abstractmethod
    def apply_with_params(self, sample: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
        """Apply data augmentation to sample with parameters."""
        raise NotImplementedError

    def __call__(self, sample: dict[str, Any]) -> dict[str, Any]:
        """Apply the transform to the input sample."""
        params = self.get_params()
        params_dependent_on_data = self.get_params_on_sample(sample, params)
        params.update(params_dependent_on_data)
        self._params = params

        if random.random() < self.p:
            return self.apply_with_params(sample, params)
        return sample

    def get_base_init_args(self) -> dict[str, Any]:
        """Returns base init args - p"""
        return {"p": self.p}


class Compose:
    """
    A class for composing multiple image transformations.

    Attributes:
        transforms (List[Callable]): A list of transformation functions to be applied sequentially.
        p (float): The probability of applying all the transforms.

    Methods:
        __call__: Applies a series of transformations to input data.
        append: Appends a new transform to the existing list of transforms.
        insert: Inserts a new transform at a specified index in the list of transforms.
        __getitem__: Retrieves a specific transform or a set of transforms using indexing.
        __setitem__: Sets a specific transform or a set of transforms using indexing.
        tolist: Converts the list of transforms to a standard Python list.

    Examples:
        >>> transforms = [RandomFlip(), RandomPerspective(30)]
        >>> compose = Compose(transforms)
        >>> transformed_data = compose(data)
        >>> compose.append(CenterCrop((224, 224)))
        >>> compose.insert(0, RandomFlip())
    """

    def __init__(self, transforms: list[TransformBase], p: float = 1):
        """
        Initializes the Compose object with a list of transforms.

        Args:
            transforms (List[TransformBase]): A list of callable transform objects to be applied sequentially.
            p (float): The probability of applying all the transforms.

        Examples:
            >>> from ultralytics.data.augment import Compose, RandomHSV, RandomFlip
            >>> transforms = [RandomHSV(), RandomFlip()]
            >>> compose = Compose(transforms)
        """
        self.p = p
        self.transforms = transforms if isinstance(transforms, list) else [transforms]

    def __call__(self, sample: dict[str, Any]) -> dict[str, Any]:
        """
        Applies a series of transformations to input data. This method sequentially applies each transformation in the
        Compose object's list of transforms to the input data.

        Args:
            sample (dict[str, Any]): The input sample to be transformed.

        Returns:
            (dict[str, Any]): The transformed data after applying all transformations in sequence.

        Examples:
            >>> transforms = [Transform1(), Transform2(), Transform3()]
            >>> compose = Compose(transforms)
            >>> transformed_data = compose(input_data)
        """
        if random.random() < self.p:
            for t in self.transforms:
                sample = t(sample)
        return sample

    def append(self, transform: TransformBase):
        """
        Appends a new transform to the existing list of transforms.

        Args:
            transform (TransformBase): The transformation to be added to the composition.

        Examples:
            >>> compose = Compose([RandomFlip(), RandomPerspective()])
            >>> compose.append(RandomHSV())
        """
        self.transforms.append(transform)

    def insert(self, index: int, transform: TransformBase):
        """
        Inserts a new transform at a specified index in the existing list of transforms.

        Args:
            index (int): The index at which to insert the new transform.
            transform (TransformBase): The transform object to be inserted.

        Examples:
            >>> compose = Compose([Transform1(), Transform2()])
            >>> compose.insert(1, Transform3())
            >>> len(compose.transforms)
            3
        """
        self.transforms.insert(index, transform)

    def __getitem__(self, index: Union[list, int]) -> "Compose":
        """
        Retrieves a specific transform or a set of transforms using indexing.

        Args:
            index (int | List[int]): Index or list of indices of the transforms to retrieve.

        Returns:
            (Compose): A new Compose object containing the selected transform(s).

        Raises:
            AssertionError: If the index is not of type int or list.

        Examples:
            >>> transforms = [RandomFlip(), RandomPerspective(10), RandomHSV(0.5, 0.5, 0.5)]
            >>> compose = Compose(transforms)
            >>> single_transform = compose[1]  # Returns a Compose object with only RandomPerspective
            >>> multiple_transforms = compose[0:2]  # Returns a Compose object with RandomFlip and RandomPerspective
        """
        assert isinstance(index, (int, list)), f"The indices should be either list or int type but got {type(index)}"
        index = [index] if isinstance(index, int) else index
        return Compose([self.transforms[i] for i in index])

    def __setitem__(self, index: Union[list, int], value: Union[list, int]) -> None:
        """
        Sets one or more transforms in the composition using indexing.

        Args:
            index (int | List[int]): Index or list of indices to set transforms at.
            value (Any | List[Any]): Transform or list of transforms to set at the specified index(es).

        Raises:
            AssertionError: If index type is invalid, value type doesn't match index type, or index is out of range.

        Examples:
            >>> compose = Compose([Transform1(), Transform2(), Transform3()])
            >>> compose[1] = NewTransform()  # Replace second transform
            >>> compose[0:2] = [NewTransform1(), NewTransform2()]  # Replace first two transforms
        """
        assert isinstance(index, (int, list)), f"The indices should be either list or int type but got {type(index)}"
        if isinstance(index, list):
            assert isinstance(value, list), (
                f"The indices should be the same type as values, but got {type(index)} and {type(value)}"
            )
        if isinstance(index, int):
            index, value = [index], [value]
        for i, v in zip(index, value):
            assert i < len(self.transforms), f"list index {i} out of range {len(self.transforms)}."
            self.transforms[i] = v

    def tolist(self) -> list[TransformBase]:
        """
        Converts the list of transforms to a standard Python list.

        Returns:
            (List): A list containing all the transform objects in the Compose instance.

        Examples:
            >>> transforms = [RandomFlip(), RandomPerspective(10), CenterCrop()]
            >>> compose = Compose(transforms)
            >>> transform_list = compose.tolist()
            >>> print(len(transform_list))
            3
        """
        return self.transforms

    def __repr__(self) -> str:
        """
        Returns a string representation of the Compose object.

        Returns:
            (str): A string representation of the Compose object, including the list of transforms.

        Examples:
            >>> transforms = [RandomFlip(), RandomPerspective(degrees=10, translate=0.1, scale=0.1)]
            >>> compose = Compose(transforms)
            >>> print(compose)
            Compose([
                RandomFlip(),
                RandomPerspective(degrees=10, translate=0.1, scale=0.1)
            ])
        """
        return f"{self.__class__.__name__}({', '.join([f'{t}' for t in self.transforms])})"
