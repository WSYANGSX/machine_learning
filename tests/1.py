# load data
            if dt is None:  # not cached in RAM
                if dtfn.exists():  # cached in Disk
                    try:
                        dt = np.load(dtfn)
                    except Exception as e:
                        LOGGER.warning(f"Removing corrupt *.npy image file {dtfn} due to: {e}")
                        Path(dtfn).unlink(missing_ok=True)
                        dt = DatasetBase.file_read(dtf)
                else:
                    dt = DatasetBase.file_read(dtf)

                data[self.modal_mapping[name]] = np.array([])
                
                
                else:
                    h0, w0 = dt.shape[:2]  # orig hw
                    if rect_mode:  # resize long side to imgsz while maintaining aspect ratio
                        r = self.imgsz / max(h0, w0)  # ratio
                        if r != 1:  # if sizes are not equal
                            w, h = (min(math.ceil(w0 * r), self.imgsz), min(math.ceil(h0 * r), self.imgsz))
                            dt = cv2.resize(dt, (w, h), interpolation=cv2.INTER_LINEAR)
                    else:
                        if not (h0 == w0 == self.imgsz):  # resize by stretching image to square imgsz
                            dt = cv2.resize(dt, (self.imgsz, self.imgsz), interpolation=cv2.INTER_LINEAR)

                    # Add to buffer if training with augmentations
                    if self.augment:
                        self.data[name][i], self.hw0[i], self.hw[i] = dt, (h0, w0), dt.shape[:2]
                        self.mosaic_buffer.append(i)
                        if 1 < len(self.mosaic_buffer) >= self.max_mosaic_buffer_length:  # prevent empty buffer
                            j = self.mosaic_buffer.pop(0)
                            if self.cache != "ram":
                                self.data[name][j], self.hw0[j], self.hw[j] = None, None, None

                    data[self.modal_mapping[name]] = dt
                    hw0 = (h0, w0)
                    hw = dt.shape[:2]
