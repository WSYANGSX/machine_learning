from typing import Union, Any

import random


class TransformBase:
    """
    Base class for image transformations.

    This class serves as a foundation for implementing various image processing operations, designed to be
    compatible with both classification and semantic segmentation tasks.

    Methods:
        apply_image: Applies image transformations to a sample.
        apply_instances: Applies transformations to object instances in a sample.
        apply_semantic: Applies semantic segmentation to an image.
        __call__: Applies all transformations to an image, instances, and semantic masks.

    Examples:
        >>> transform = BaseTransform()
        >>> sample = {"image": np.array(...), "instances": [...], "semantic": np.array(...)}
        >>> transformed_sample = transform(sample)
    """

    def __init__(self) -> None:
        """
        Initializes the BaseTransform object.

        This constructor sets up the base transformation object, which can be extended for specific image
        processing tasks. It is designed to be compatible with both classification and semantic segmentation.

        Examples:
            >>> transform = BaseTransform()
        """
        pass

    def apply_image(self, sample: Any):
        """
        Applies image transformations to sample.

        This method is intended to be overridden by subclasses to implement specific image transformation
        logic. In its base form, it returns the input sample unchanged.

        Args:
            sample (Any): The input sample to be transformed. The exact type and structure of sample may
                vary depending on the specific implementation.

        Returns:
            (Any): The transformed sample. In the base implementation, this is identical to the input.

        Examples:
            >>> transform = BaseTransform()
            >>> original_sample = [1, 2, 3]
            >>> transformed_sample = transform.apply_image(original_sample)
            >>> print(transformed_sample)
            [1, 2, 3]
        """
        pass

    def apply_instances(self, sample: dict[str, Any]):
        """
        Applies transformations to object instances in sample.

        This method is responsible for applying various transformations to object instances within the given
        sample. It is designed to be overridden by subclasses to implement specific instance transformation
        logic.

        Args:
            sample (dict[str, Any]): A dictionary containing label information, including object instances.

        Returns:
            (dict[str, Any]): The modified sample dictionary with transformed object instances.

        Examples:
            >>> transform = BaseTransform()
            >>> sample = {"instances": Instances(xyxy=torch.rand(5, 4), cls=torch.randint(0, 80, (5,)))}
            >>> transformed_sample = transform.apply_instances(sample)
        """
        pass

    def apply_semantic(self, sample: Any):
        """
        Applies semantic segmentation transformations to an image.

        This method is intended to be overridden by subclasses to implement specific semantic segmentation
        transformations. In its base form, it does not perform any operations.

        Args:
            sample (Any): The input sample or semantic segmentation mask to be transformed.

        Returns:
            (Any): The transformed semantic segmentation mask or sample.

        Examples:
            >>> transform = BaseTransform()
            >>> semantic_mask = np.zeros((100, 100), dtype=np.uint8)
            >>> transformed_mask = transform.apply_semantic(semantic_mask)
        """
        pass

    def __call__(self, sample: dict[str, Any]) -> dict[str, Any]:
        """
        Applies all sample transformations to an image, instances, and semantic masks.

        This method orchestrates the application of various transformations defined in the BaseTransform class
        to the input sample. It sequentially calls the apply_image and apply_instances methods to process the
        image and object instances, respectively.

        Args:
            sample (dict[str, Any]): A dictionary containing image data and annotations. Expected keys include 'img' for
                the image data, and 'instances' for object instances.

        Returns:
            (dict[str, Any]): The transformed sample dictionary with image and instances.

        Examples:
            >>> transform = BaseTransform()
            >>> sample = {"img": np.random.rand(640, 640, 3), "instances": []}
            >>> transformed_sample = transform(sample)
        """
        self.apply_image(sample)
        self.apply_instances(sample)
        self.apply_semantic(sample)


class MixTransformBase:
    """
    Base class for mix transformations like MixUp and Mosaic.

    This class provides a foundation for implementing mix transformations on datasets. It handles the
    probability-based application of transforms and manages the mixing of multiple images and labels.

    Attributes:
        dataset (Any): The dataset object containing images and labels.
        pre_transform (Callable | None): Optional transform to apply before mixing.
        p (float): Probability of applying the mix transformation.

    Methods:
        __call__: Applies the mix transformation to the input sample.
        _mix_transform: Abstract method to be implemented by subclasses for specific mix operations.
        get_indexes: Abstract method to get indexes of images to be mixed.
        _update_label_text: Updates label text for mixed images.

    Examples:
        >>> class CustomMixTransform(BaseMixTransform):
        ...     def _mix_transform(self, sample):
        ...         # Implement custom mix logic here
        ...         return sample
        ...
        ...     def get_indexes(self):
        ...         return [random.randint(0, len(self.dataset) - 1) for _ in range(3)]
        >>> dataset = YourDataset()
        >>> transform = CustomMixTransform(dataset, p=0.5)
        >>> mixed_sample = transform(original_sample)
    """

    def __init__(self, dataset, pre_transform=None, p=0.0) -> None:
        """
        Initializes the BaseMixTransform object for mix transformations like MixUp and Mosaic.

        This class serves as a base for implementing mix transformations in image processing pipelines.

        Args:
            dataset (Any): The dataset object containing images and labels for mixing.
            pre_transform (Callable | None): Optional transform to apply before mixing.
            p (float): Probability of applying the mix transformation. Should be in the range [0.0, 1.0].

        Examples:
            >>> dataset = YOLODataset("path/to/data")
            >>> pre_transform = Compose([RandomFlip(), RandomPerspective()])
            >>> mix_transform = BaseMixTransform(dataset, pre_transform, p=0.5)
        """
        self.dataset = dataset
        self.pre_transform = pre_transform
        self.p = p

    def __call__(self, sample):
        """
        Applies pre-processing transforms and mixup/mosaic transforms to sampled data.

        This method determines whether to apply the mix transform based on a probability factor. If applied, it
        selects additional images, applies pre-transforms if specified, and then performs the mix transform.

        Args:
            sample (Dict): A dictionary containing label data and an image.

        Returns:
            (Dict): The transformed labels dictionary, which may include mixed data from other images.

        Examples:
            >>> transform = BaseMixTransform(dataset, pre_transform=None, p=0.5)
            >>> result = transform({"image": img, "bboxes": boxes, "cls": classes})
        """
        if random.uniform(0, 1) > self.p:
            return sample

        # Get index of one or three other images
        indexes = self.get_indexes()
        if isinstance(indexes, int):
            indexes = [indexes]

        # Get images information will be used for Mosaic or MixUp
        mix_samples = [self.dataset.get_sample(i) for i in indexes]

        if self.pre_transform is not None:
            for i, data in enumerate(mix_samples):
                mix_samples[i] = self.pre_transform(data)
        sample["mix_samples"] = mix_samples

        # Update cls and texts
        sample = self._update_sample_text(sample)
        # Mosaic or MixUp
        sample = self._mix_transform(sample)
        sample.pop("mix_samples", None)
        return sample

    def _mix_transform(self, sample):
        """
        Applies MixUp or Mosaic augmentation to the sample dictionary.

        This method should be implemented by subclasses to perform specific mix transformations like MixUp or
        Mosaic. It modifies the input sample dictionary in-place with the augmented data.

        Args:
            sample (Dict): A dictionary containing image and label data. Expected to have a 'mix_labels' key
                with a list of additional image and label data for mixing.

        Returns:
            (Dict): The modified sample dictionary with augmented data after applying the mix transform.

        Examples:
            >>> transform = BaseMixTransform(dataset)
            >>> sample = {"image": img, "bboxes": boxes, "mix_labels": [{"image": img2, "bboxes": boxes2}]}
            >>> augmented_sample = transform._mix_transform(sample)
        """
        raise NotImplementedError

    def get_indexes(self):
        """
        Gets a list of shuffled indexes for mosaic augmentation.

        Returns:
            (List[int]): A list of shuffled indexes from the dataset.

        Examples:
            >>> transform = BaseMixTransform(dataset)
            >>> indexes = transform.get_indexes()
            >>> print(indexes)  # [3, 18, 7, 2]
        """
        raise NotImplementedError

    @staticmethod
    def _update_sample_text(sample: dict[str, Any]) -> dict[str, Any]:
        """
        Updates label text and class IDs for mixed samples in a sample.

        This method processes the 'texts' and 'cls' fields of the input sample dictionary and any mixed samples,
        creating a unified sample.

        Args:
            sample (dict[str, Any]): A dictionary containing label information, including 'texts' and 'cls' fields,
                and optionally a 'mix_samples' field with additional sample dictionaries.

        Returns:
            (dict[str, Any]): The updated sample dictionary with unified text labels and updated class IDs.

        Examples:
            >>> sample = {
            ...     "texts": [["cat"], ["dog"]],
            ...     "cls": torch.tensor([[0], [1]]),
            ...     "mix_samples": [{"texts": [["bird"], ["fish"]], "cls": torch.tensor([[0], [1]])}],
            ... }
            >>> updated_sample = self._update_sample_text(sample)
            >>> print(updated_sample["texts"])
            [['cat'], ['dog'], ['bird'], ['fish']]
            >>> print(updated_sample["cls"])
            tensor([[0],
                    [1]])
            >>> print(updated_sample["mix_samples"][0]["cls"])
            tensor([[2],
                    [3]])
        """
        if "texts" not in sample:
            return sample

        mix_texts = sum([sample["texts"]] + [x["texts"] for x in sample["mix_samples"]], [])
        mix_texts = list({tuple(x) for x in mix_texts})
        text2id = {text: i for i, text in enumerate(mix_texts)}

        for item in [sample] + sample["mix_samples"]:
            for i, cls in enumerate(item["cls"].squeeze(-1).tolist()):
                text = item["texts"][int(cls)]
                item["cls"][i] = text2id[tuple(text)]
            item["texts"] = mix_texts
        return sample


class Compose:
    """
    A class for composing multiple image transformations.

    Attributes:
        transforms (List[Callable]): A list of transformation functions to be applied sequentially.

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

    def __init__(self, transforms):
        """
        Initializes the Compose object with a list of transforms.

        Args:
            transforms (List[Callable]): A list of callable transform objects to be applied sequentially.

        Examples:
            >>> from ultralytics.data.augment import Compose, RandomHSV, RandomFlip
            >>> transforms = [RandomHSV(), RandomFlip()]
            >>> compose = Compose(transforms)
        """
        self.transforms = transforms if isinstance(transforms, list) else [transforms]

    def __call__(self, data):
        """
        Applies a series of transformations to input data. This method sequentially applies each transformation in the
        Compose object's list of transforms to the input data.

        Args:
            data (Any): The input data to be transformed. This can be of any type, depending on the
                transformations in the list.

        Returns:
            (Any): The transformed data after applying all transformations in sequence.

        Examples:
            >>> transforms = [Transform1(), Transform2(), Transform3()]
            >>> compose = Compose(transforms)
            >>> transformed_data = compose(input_data)
        """
        for t in self.transforms:
            data = t(data)
        return data

    def append(self, transform):
        """
        Appends a new transform to the existing list of transforms.

        Args:
            transform (BaseTransform): The transformation to be added to the composition.

        Examples:
            >>> compose = Compose([RandomFlip(), RandomPerspective()])
            >>> compose.append(RandomHSV())
        """
        self.transforms.append(transform)

    def insert(self, index, transform):
        """
        Inserts a new transform at a specified index in the existing list of transforms.

        Args:
            index (int): The index at which to insert the new transform.
            transform (BaseTransform): The transform object to be inserted.

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

    def tolist(self):
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

    def __repr__(self):
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
