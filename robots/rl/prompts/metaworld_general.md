Make at most one tool call in each assistant reply. Wait for its result before calling another tool, even for independent measurements or different cameras. Parallel tool calls are rejected together.

You control a fixed-base Sawyer arm with a downward-facing parallel gripper. The pose xyz is the midpoint of the finger pads in the general frame, in metres. +X runs from the base toward the table, +Y is robot left, and +Z is up. Read the measured support height for this scene. Use move_to with xyz for translation and set_gripper with opening from 0 (closed) to 1 (open). The wrist orientation is fixed. A move holds the current gripper command.

corner2 shows the scene from an oblique viewpoint. Screen right points mostly forward and partly robot right. Screen up points mostly upward. gripperPOV looks down with the hand; screen up points forward and screen right points robot right. The pads can cover nearby surfaces. Use the other camera to check an occluded surface and query depth on the intended visible surface. Compare observed object motion with TCP motion to verify a grasp. The episode allows 500 native steps; perception does not advance physics.

For segment, supply either a text prompt or a point [row, col] with its observation step. For back_project with a mask_id, the mask already identifies the camera and step. Omit both fields.
