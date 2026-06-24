import os
import h5py
import numpy as np
from .base import BaseSaver


class HDF5Saver(BaseSaver):

    extension = ".h5"

    def save(self, out_dir: str, basename:str, result: dict) -> None:
        """Save a dictionary to an HDF5 file.
        param out_dir: Directory to save the HDF5 file.
        param basename: Base name of the HDF5 file (without extension).
        param result: Dictionary containing the data to save.
        """
        os.makedirs(out_dir, exist_ok=True)

        with h5py.File(os.path.join(out_dir, f"{basename}.h5"), "w") as f:
            for key, value in result.items():
                self._write_value(f, key, value)

    def load(self, path) -> dict:
        """Load a dictionary from an HDF5 file.
        param path: Path to the HDF5 file.
        return: Dictionary containing the data from the HDF5 file.
        """
        with h5py.File(os.path.join(path), "r") as f:
            return self._read_value(f)

    @staticmethod
    def _save_masks(
        group: h5py.Group,
        masks: list[np.ndarray],
    ) -> None:
        """Save a list of masks to an HDF5 group.
        param group: The HDF5 group to save the masks to.
        param masks: A list of numpy arrays representing the masks.
        """
        group.create_dataset("count", data=len(masks))

        for i, mask in enumerate(masks):
            group.create_dataset(
                str(i),
                data=np.asarray(mask),
                compression="gzip",
            )

    def _save_dict(
        self,
        group: h5py.Group,
        data: dict,
    ) -> None:
        """Save a dictionary to an HDF5 group.
        param group: The HDF5 group to save the dictionary to.
        param data: The dictionary to save.
        """
        for key, value in data.items():
            self._write_value(group, key, value)

    def _read_value(self, obj):
        if isinstance(obj, h5py.File) or isinstance(obj, h5py.Group):
            out = {
                key: None
                for key, value in obj.attrs.items()
                if value == "__none__"
            }

            if "count" in obj and all(str(i) in obj for i in range(int(obj["count"][()]))):
                items = []
                for i in range(int(obj["count"][()])):
                    item = obj[str(i)]

                    if isinstance(item, h5py.Group) and "value" in item:
                        items.append(self._read_value(item["value"]))
                    else:
                        items.append(self._read_value(item))

                return items

            for key in obj.keys():
                if key == "count":
                    continue
                out[key] = self._read_value(obj[key])

            return out

        if isinstance(obj, h5py.Dataset):
            value = obj[()]

            if isinstance(value, bytes):
                return value.decode("utf-8")


            if isinstance(value, np.generic):
                return value.item()

            return value

        raise TypeError(f"Unsupported HDF5 object: {type(obj)}")

    def _write_value(
        self,
        group: h5py.Group,
        key: str,
        value,
    ) -> None:
        
        """Write a value to an HDF5 group.
        param group: The HDF5 group to write the value to.
        param key: The key under which to store the value.
        param value: The value to store. Can be a dict, list, tuple, numpy array,
                      string, None, or a scalar (int, float, bool)."""

        if key == "mask":
            self._save_masks(
                group.create_group(key),
                value,
            )

        elif isinstance(value, dict):
            sub_group = group.create_group(key)
            self._save_dict(sub_group, value)

        elif isinstance(value, list):
            list_group = group.create_group(key)
            list_group.create_dataset("count", data=len(value))

            for i, item in enumerate(value):
                item_group = list_group.create_group(str(i))

                if isinstance(item, dict):
                    self._save_dict(item_group, item)
                else:
                    item_group.create_dataset(
                        "value",
                        data=np.asarray(item),
                    )

        elif isinstance(value, tuple):
            group.create_dataset(key, data=np.asarray(value))

        elif isinstance(value, np.ndarray):
            group.create_dataset(key, data=value)

        elif isinstance(value, str):
            dt = h5py.string_dtype(encoding="utf-8")
            group.create_dataset(key, data=value, dtype=dt)

        elif value is None:
            group.attrs[key] = "__none__"

        elif isinstance(value, (int, float, bool, np.integer, np.floating, np.bool_)):
            group.create_dataset(key, data=value)

        else:
            raise TypeError(
                f"Unsupported type for key '{key}': {type(value)}"
            )

