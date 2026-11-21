#!/usr/bin/env python3

import traceback

import cv2
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy

from sensor_msgs.msg import Image
from std_msgs.msg import Bool, Float32
from cv_bridge import CvBridge


class LaneDetectorNode(Node):
    def __init__(self):
        super().__init__("lane_detector_node")

        self.bridge = CvBridge()
        self.draw_degree = 1
        # ================= Parameters =================
        self.camera_topic = "/camera/image_raw"

        self.frame_width = 320
        self.frame_height = 180


        self.publish_width = 640
        self.publish_height = 360


        self.process_every_n_frames = 3
        self._frame_counter = 0

        self.crop_top_ratio = 0.6
        self.roi_top_ratio = 0.05


        self.white_v_min = 150
        self.white_s_max = 30

        self.num_scan_lines = 25
        self.scan_band_height = 10
        self.min_pixels_per_band = 10
        self.green_alpha = 0.18

       
        self.min_lane_line_height_px = 40

        
        self.min_obstacle_area_px = 3

        
        self.cluster_gap_px = 20

        
        self.min_gap_width_px = 40

        
        self.max_useful_gap_width_px = 520


        self.continuity_weight = 0.6


        self.min_center_points_for_detection = 5


        self.prev_path_center_x = None

        # ================= ROS =================
        qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10
        )

        self.sub = self.create_subscription(
            Image,
            self.camera_topic,
            self.image_callback,
            qos
        )

        self.debug_pub = self.create_publisher(
            Image,
            "/lane/debug_image",
            10
        )

        self.mask_pub = self.create_publisher(
            Image,
            "/lane/white_mask",
            10
        )

        self.detected_pub = self.create_publisher(
            Bool,
            "/lane/detected",
            10
        )

        self.error_pub = self.create_publisher(
            Float32,
            "/lane/center_error_px",
            10
        )

        self.prev_left_fit = None
        self.prev_right_fit = None

        self.get_logger().info(f"Lane detector subscribed to {self.camera_topic}")

    @staticmethod
    def _make_image_msg(cv_image, encoding, header):
        """Build a sensor_msgs/Image message directly, bypassing
        cv_bridge's cv2_to_imgmsg(). On this system that function's
        internal cvtype_to_name lookup table is missing the entry for
        type code 16 (CV_8UC3 - any standard 3-channel 8-bit image), so
        ANY attempt to publish a bgr8/rgb8 image through it raises
        KeyError: 16, no matter what encoding string is passed. This
        produces the exact same wire format cv_bridge would, without
        going through that broken code path at all."""
        img_msg = Image()
        img_msg.header = header
        img_msg.height = cv_image.shape[0]
        img_msg.width = cv_image.shape[1]
        img_msg.encoding = encoding
        img_msg.is_bigendian = 0
        channels = 1 if cv_image.ndim == 2 else cv_image.shape[2]
        img_msg.step = cv_image.shape[1] * channels * cv_image.itemsize
        img_msg.data = cv_image.tobytes()
        return img_msg

    def get_white_mask(self, frame_bgr):
        # HSV channel order in OpenCV is (Hue, Saturation, Value) -
        # index 0 = H, index 1 = S, index 2 = V. Checking BOTH value
        # (brightness) and saturation (colorfulness) is what makes this
        # "white only" - grayscale brightness alone can't tell white
        # apart from any other bright color.
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        s_channel = hsv[:, :, 1]  # Saturation
        v_channel = hsv[:, :, 2]  # Value (brightness)

        # Slight blur first to reduce pixel-level noise before thresholding
        v_channel = cv2.GaussianBlur(v_channel, (5, 5), 0)

        mask = np.zeros_like(v_channel, dtype=np.uint8)
        white_condition = (
            (v_channel >= self.white_v_min) &
            (s_channel <= self.white_s_max)
        )
        mask[white_condition] = 255

        # Remove tiny blobs (cheap erode/dilate instead of open/close
        # with a larger kernel - kept from your edit for speed)
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.erode(mask, kernel, iterations=1)
        mask = cv2.dilate(mask, kernel, iterations=1)

        return mask

    def apply_roi(self, mask):

        h, w = mask.shape

        roi_mask = np.zeros_like(mask)

        y_top = int(self.roi_top_ratio * h)

        polygon = np.array([[
            (0, h),
            (0, y_top),
            (w, y_top),
            (w, h)
        ]], dtype=np.int32)

        cv2.fillPoly(roi_mask, polygon, 255)

        return cv2.bitwise_and(mask, roi_mask)

    def _classify_blobs(self, mask):
        """Run connected components on the FULL mask (not per-row) and
        split blobs into lane boundaries (tall/long - the real tape
        lines) vs obstacles (everything else white, however small).
        Returns (left_pts, right_pts, obstacle_blobs) where left_pts /
        right_pts are (xs, ys) arrays for the merged left/right lane
        blobs (or None if that side has no qualifying blob), and
        obstacle_blobs is a list of (xs, ys) arrays."""

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            mask, connectivity=8
        )

        h, w = mask.shape
        image_center_x = w / 2.0

        lane_blob_candidates = []  # list of (mean_x, xs, ys)
        obstacle_blobs = []

        for label in range(1, num_labels):
            area = stats[label, cv2.CC_STAT_AREA]
            if area < self.min_obstacle_area_px:
                continue  # true single-pixel noise, ignore entirely

            bh = stats[label, cv2.CC_STAT_HEIGHT]
            ys, xs = np.where(labels == label)

            if bh >= self.min_lane_line_height_px:
                lane_blob_candidates.append((float(np.mean(xs)), xs, ys))
            else:
                # A real obstacle (per your floor spots) - keep it,
                # however small, as long as it clears the noise floor.
                obstacle_blobs.append((xs, ys))

        left_xs_list, left_ys_list = [], []
        right_xs_list, right_ys_list = [], []

        for mean_x, xs, ys in lane_blob_candidates:
            if mean_x < image_center_x:
                left_xs_list.append(xs)
                left_ys_list.append(ys)
            else:
                right_xs_list.append(xs)
                right_ys_list.append(ys)

        # Merge same-side blobs (handles dashed/segmented lane lines -
        # each dash may be its own connected component)
        left_pts = None
        if left_xs_list:
            left_pts = (np.concatenate(left_xs_list), np.concatenate(left_ys_list))

        right_pts = None
        if right_xs_list:
            right_pts = (np.concatenate(right_xs_list), np.concatenate(right_ys_list))

        return left_pts, right_pts, obstacle_blobs

    def _fit_boundary(self, pts):
        """Fit x = f(y) (degree 1: straight line) to a lane boundary's
        pixel coordinates so we get a continuous edge even across gaps
        in a dashed line. Returns (poly1d, y_min, y_max) or None.

        y_min/y_max are the actual observed range of this blob - the
        caller MUST clamp any y it evaluates at to this range rather
        than plugging in an arbitrary y. Without clamping, a line that
        only exists in the bottom part of frame gets its slope
        extrapolated far past where it actually is, which can make left
        and right boundaries mathematically converge/cross near the top
        of frame - producing a nonsense near-zero-width "corridor" where
        no real information exists.
        """

        if pts is None:
            return None

        xs, ys = pts
        if len(np.unique(ys)) < 2:
            return None  # not enough vertical spread to fit a line

        coeffs = np.polyfit(ys, xs, deg=1)
        return np.poly1d(coeffs), int(ys.min()), int(ys.max())

    def _find_blobs(self, xs_sorted):
        """Group sorted, unique x-coordinates into blobs (contiguous
        white regions), splitting wherever consecutive x's are farther
        apart than cluster_gap_px. Returns a list of (start_x, end_x)."""

        blobs = []
        blob_start = xs_sorted[0]
        prev_x = xs_sorted[0]

        for x in xs_sorted[1:]:
            if x - prev_x > self.cluster_gap_px:
                blobs.append((blob_start, prev_x))
                blob_start = x
            prev_x = x

        blobs.append((blob_start, prev_x))
        return blobs

    def _find_gaps(self, blobs, left_bound, right_bound):
        """Given obstacle blobs (sorted left-to-right) that fall inside
        [left_bound, right_bound] (the current lane corridor), return
        candidate safe gaps: corridor-edge -> first blob, blob -> next
        blob, last blob -> corridor-edge. NEVER extends past left_bound/
        right_bound - that's what keeps the path from crossing the lane."""

        if not blobs:
            return [(left_bound, right_bound)]

        gaps = []

        first_start, _ = blobs[0]
        if first_start > left_bound:
            gaps.append((left_bound, first_start))

        for i in range(len(blobs) - 1):
            _, prev_end = blobs[i]
            next_start, _ = blobs[i + 1]
            if next_start - prev_end >= self.min_gap_width_px:
                gaps.append((prev_end, next_start))

        _, last_end = blobs[-1]
        if last_end < right_bound:
            gaps.append((last_end, right_bound))

        return gaps

    def _select_best_gap(self, gaps, prev_center_x):
        """Score each candidate gap by (capped) width minus a penalty for
        drifting away from the previous known-good path position, and
        return the best one's center x. This is what makes the chosen
        gap prefer staying near lane center over hugging the widest side
        of an obstacle, while staying continuous frame-to-frame."""

        best_center = None
        best_score = -float("inf")

        for (g0, g1) in gaps:
            width = g1 - g0
            if width < self.min_gap_width_px:
                continue

            center = (g0 + g1) / 2.0
            width_score = min(width, self.max_useful_gap_width_px)
            continuity_penalty = self.continuity_weight * abs(center - prev_center_x)
            score = width_score - continuity_penalty

            if score > best_score:
                best_score = score
                best_center = center

        return best_center

    def fit_lanes(self, mask):

        h, w = mask.shape
        image_center_x = w / 2.0

        left_pts, right_pts, obstacle_blobs = self._classify_blobs(mask)

        if left_pts is None and right_pts is None:
            # No qualifying lane boundary found anywhere in frame - too
            # little information to trust a corridor. Don't guess.
            return (
                np.array([], dtype=np.int32).reshape(0, 2),
                np.array([], dtype=np.int32).reshape(0, 2),
                np.array([], dtype=np.int32).reshape(0, 2),
                False,
                0.0,
            )

        left_fit = self._fit_boundary(left_pts)
        right_fit = self._fit_boundary(right_pts)

        y_start = int(self.roi_top_ratio * h)
        y_end = h - 10
        y_values = np.linspace(y_start, y_end, self.num_scan_lines).astype(int)

        left_points = []
        right_points = []
        center_points = []

        if self.prev_path_center_x is not None:
            running_center_x = self.prev_path_center_x
        else:
            running_center_x = image_center_x

        for y in y_values:
            # Corridor bounds at this row - from the fitted lane boundary
            # if we have one on that side, clamped to where that blob
            # actually has data (never extrapolated past it), otherwise
            # the frame edge (conservative: treat unknown side as wide
            # open).
            if left_fit is not None:
                lf, ly_min, ly_max = left_fit
                left_bound = float(np.clip(lf(np.clip(y, ly_min, ly_max)), 0, w))
            else:
                left_bound = 0.0

            if right_fit is not None:
                rf, ry_min, ry_max = right_fit
                right_bound = float(np.clip(rf(np.clip(y, ry_min, ry_max)), 0, w))
            else:
                right_bound = float(w)

            if left_bound >= right_bound:
                # Degenerate/crossed corridor at this row - skip, don't guess
                continue

            left_points.append([int(left_bound), y])
            right_points.append([int(right_bound), y])

            # Obstacles whose pixels fall within this row's band AND
            # within the current corridor
            y1 = max(0, y - self.scan_band_height // 2)
            y2 = min(h, y + self.scan_band_height // 2)

            obstacle_xs_in_band = []
            for xs, ys in obstacle_blobs:
                row_mask = (ys >= y1) & (ys < y2)
                if not np.any(row_mask):
                    continue
                xs_in_row = xs[row_mask]
                xs_in_row = xs_in_row[(xs_in_row >= left_bound) & (xs_in_row <= right_bound)]
                if len(xs_in_row) > 0:
                    obstacle_xs_in_band.append(xs_in_row)

            if obstacle_xs_in_band:
                xs_sorted = np.unique(np.concatenate(obstacle_xs_in_band))
                obstacle_group_blobs = self._find_blobs(xs_sorted)
            else:
                obstacle_group_blobs = []

            gaps = self._find_gaps(obstacle_group_blobs, left_bound, right_bound)
            gap_center_x = self._select_best_gap(gaps, running_center_x)

            if gap_center_x is None:
                # Obstacle(s) block the entire corridor width at this
                # row - no safe gap through, skip (don't guess).
                continue

            center_points.append([int(gap_center_x), y])
            running_center_x = gap_center_x

        left_points = np.array(left_points, dtype=np.int32)
        right_points = np.array(right_points, dtype=np.int32)
        center_points = np.array(center_points, dtype=np.int32)

        lane_detected = len(center_points) >= self.min_center_points_for_detection

        if lane_detected:
            bottom_center_x = center_points[-1][0]
            center_error_px = float(bottom_center_x - image_center_x)
            self.prev_path_center_x = float(bottom_center_x)
        else:
            center_error_px = 0.0

        return left_points, right_points, center_points, lane_detected, center_error_px

    def draw_result(
        self,
        frame_bgr,
        mask,
        left_points,
        right_points,
        center_points,
        lane_detected,
        center_error_px
    ):
        result = frame_bgr.copy()
        h, w = frame_bgr.shape[:2]

        def prepare_points(points):

            if points is None or len(points) < 2:
                return None

            points = np.array(points, dtype=np.int32)
            points = points[np.argsort(points[:, 1])]

            xs = points[:, 0].astype(np.float32)
            ys = points[:, 1].astype(np.float32)

            if len(xs) >= 5:
                kernel_size = 5
                kernel = np.ones(kernel_size) / kernel_size
                xs_smooth = np.convolve(xs, kernel, mode="same")

                xs_smooth[0] = xs[0]
                xs_smooth[1] = xs[1]
                xs_smooth[-1] = xs[-1]
                xs_smooth[-2] = xs[-2]
            else:
                xs_smooth = xs

            smooth_points = np.array(
                [[int(x), int(y)] for x, y in zip(xs_smooth, ys)],
                dtype=np.int32
            )

            return smooth_points

        left_smooth = prepare_points(left_points)
        right_smooth = prepare_points(right_points)

        computed_center = None

        if left_smooth is not None and right_smooth is not None:
            left_y = left_smooth[:, 1]
            left_x = left_smooth[:, 0]

            right_y = right_smooth[:, 1]
            right_x = right_smooth[:, 0]

            y_min = max(left_y.min(), right_y.min())
            y_max = min(left_y.max(), right_y.max())

            if y_max > y_min:
                y_common = np.linspace(y_min, y_max, 50)

                left_x_common = np.interp(y_common, left_y, left_x)
                right_x_common = np.interp(y_common, right_y, right_x)

                pts_left = np.array(
                    [[int(x), int(y)] for x, y in zip(left_x_common, y_common)],
                    dtype=np.int32
                )

                pts_right = np.array(
                    [[int(x), int(y)] for x, y in zip(right_x_common, y_common)],
                    dtype=np.int32
                )

                center_x_common = (left_x_common + right_x_common) / 2.0

                computed_center = np.array(
                    [[int(x), int(y)] for x, y in zip(center_x_common, y_common)],
                    dtype=np.int32
                )

                polygon_points = np.vstack((pts_left, pts_right[::-1]))

                overlay = result.copy()
                cv2.fillPoly(overlay, [polygon_points], color=(0, 255, 0))

                alpha = getattr(self, "green_alpha", 0.18)
                result = cv2.addWeighted(overlay, alpha, result, 1.0 - alpha, 0)

                cv2.polylines(
                    result,
                    [pts_left],
                    isClosed=False,
                    color=(0, 0, 255),
                    thickness=4
                )

                cv2.polylines(
                    result,
                    [pts_right],
                    isClosed=False,
                    color=(255, 0, 0),
                    thickness=4
                )

                cv2.polylines(
                    result,
                    [computed_center],
                    isClosed=False,
                    color=(0, 255, 255),
                    thickness=4
                )

                for p in computed_center[::5]:
                    cv2.circle(result, tuple(p), 4, (0, 255, 255), -1)

                bottom_center_x = computed_center[-1][0]
                image_center_x = w / 2.0
                center_error_px = float(bottom_center_x - image_center_x)
                lane_detected = True

        else:

            if left_smooth is not None:
                cv2.polylines(
                    result,
                    [left_smooth],
                    isClosed=False,
                    color=(0, 0, 255),
                    thickness=4
                )

            if right_smooth is not None:
                cv2.polylines(
                    result,
                    [right_smooth],
                    isClosed=False,
                    color=(255, 0, 0),
                    thickness=4
                )

            if center_points is not None and len(center_points) >= 2:
                center_points = np.array(center_points, dtype=np.int32)
                cv2.polylines(
                    result,
                    [center_points],
                    isClosed=False,
                    color=(0, 255, 255),
                    thickness=4
                )

        # Status text
        if lane_detected:
            status = f"Lane detected | error = {center_error_px:.1f} px"
            text_color = (0, 255, 0)
        else:
            status = "Lane not fully detected"
            text_color = (0, 0, 255)

        cv2.putText(
            result,
            status,
            (30, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            text_color,
            2
        )

        return result

    def image_callback(self, msg):
        self.get_logger().info('image_callback triggered', throttle_duration_sec=1.0)

        self._frame_counter += 1
        if self._frame_counter % self.process_every_n_frames != 0:
            return

        try:
            # Gazebo's camera sensor is configured with <format>R8G8B8</format>,
            # which ros_gz_bridge publishes as ROS encoding "rgb8" (red-green-
            # blue channel order) - NOT "bgr8". Asking cv_bridge to convert
            # straight to "bgr8" was the source of the earlier cryptic
            # "Lane detection error: 16" (a raw OpenCV type code leaking out
            # of a channel-order mismatch). Fix: read it as rgb8, then do the
            # channel swap explicitly with OpenCV.
            frame_rgb = self.bridge.imgmsg_to_cv2(
                msg,
                desired_encoding="rgb8"
            )
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

            frame_bgr = cv2.resize(
                frame_bgr,
                (self.frame_width, self.frame_height)
            )

            # ============================
            # Software camera adjustment
            # ============================
            h, w = frame_bgr.shape[:2]

            crop_y = int(self.crop_top_ratio * h)

            cropped_frame = frame_bgr[crop_y:h, :].copy()

            white_mask = self.get_white_mask(cropped_frame)
            roi_mask = self.apply_roi(white_mask)

            left_points, right_points, center_points, lane_detected, center_error_px = self.fit_lanes(
                roi_mask
            )

            result_crop = self.draw_result(
                cropped_frame,
                roi_mask,
                left_points,
                right_points,
                center_points,
                lane_detected,
                center_error_px
            )

            # ============================
            # Publish outputs
            # ============================
            detected_msg = Bool()
            detected_msg.data = bool(lane_detected)
            self.detected_pub.publish(detected_msg)

            error_msg = Float32()
            error_msg.data = float(center_error_px)
            self.error_pub.publish(error_msg)

            # Downscale just for publishing/viewing, if configured smaller
            # than the processing resolution above.
            if self.publish_width is not None and self.publish_height is not None:
                publish_frame = cv2.resize(result_crop, (self.publish_width, self.publish_height))
                publish_mask = cv2.resize(roi_mask, (self.publish_width, self.publish_height))
            else:
                publish_frame = result_crop
                publish_mask = roi_mask

            debug_msg = self._make_image_msg(publish_frame, "bgr8", msg.header)
            self.debug_pub.publish(debug_msg)

            mask_msg = self._make_image_msg(publish_mask, "mono8", msg.header)
            self.mask_pub.publish(mask_msg)

            # NOTE: cv2.imshow(...) debug windows are intentionally not
            # used here. They rely on a GTK display and will crash on
            # this system the same way cv2.imshow did earlier
            # (Gtk-Message / snap libpthread symbol lookup error). View
            # /lane/debug_image and /lane/white_mask instead, via
            # web_video_server, exactly as set up previously.

        except Exception as e:
            # NOTE: traceback.format_exc() reads the CURRENTLY-handled
            # exception directly from the interpreter's exception state, so
            # it works fine here without needing "e" at all - keeping both
            # is just for a quick one-line summary before the full trace.
            self.get_logger().error(f"Lane detection error: {e}")
            self.get_logger().error(traceback.format_exc())


def main():
    rclpy.init()
    node = LaneDetectorNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()