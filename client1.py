#!/usr/bin/env python3
"""
WebRTC Client for Windows - Receives video stream from Raspberry Pi
and processes it with the 3D object detection system
"""

import asyncio
import json
import logging
import os
import sys
import time
import cv2
import numpy as np
import torch
from pathlib import Path
from collections import deque
import websockets
from aiortc import RTCPeerConnection, RTCSessionDescription, RTCIceCandidate
import threading
import queue
import traceback

# Set MPS fallback for operations not supported on Apple Silicon
if hasattr(torch, 'backends') and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
logging.getLogger('aiortc').setLevel(logging.DEBUG)

# Import your detection modules
try:
    from detection_model import ObjectDetector
    from depth_model import DepthEstimator
    from bbox3d_utils import BBox3DEstimator, BirdEyeView
    from load_camera_params import load_camera_params, apply_camera_params_to_estimator
except ImportError as e:
    logger.error(f"Failed to import detection modules: {e}")
    sys.exit(1)

class WebRTCVideoReceiver:
    """WebRTC client to receive video stream from Raspberry Pi"""
    
    def __init__(self, server_url="ws://172.20.66.142:8080/ws"):
        self.server_url = server_url
        self.pc = None
        self.frame_queue = queue.Queue(maxsize=5)
        self.running = False
        
    async def create_valid_offer(self):
        """Create a valid offer with video media line"""
        for _ in range(3):  # Retry up to 3 times
            try:
                offer = await self.pc.createOffer()
                if offer is None or offer.sdp is None or 'm=video' not in offer.sdp:
                    logger.warning("Invalid offer generated, retrying...")
                    await asyncio.sleep(0.1)
                    continue
                return offer
            except Exception as e:
                logger.error(f"Error creating offer: {e}")
        logger.error("Failed to create valid offer after retries")
        return None
    
    async def connect_and_receive(self):
        """Connect to WebRTC server and start receiving video"""
        try:
            self.pc = RTCPeerConnection()
            self.running = True
            
            # Add video transceiver
            self.pc.addTransceiver('video', direction='recvonly')
            logger.debug("Added video transceiver")
            
            @self.pc.on("track")
            def on_track(track):
                logger.info(f"Received track: {track.kind}")
                if track.kind == "video":
                    asyncio.create_task(self.process_video_track(track))
            
            @self.pc.on("connectionstatechange")
            async def on_connectionstatechange():
                logger.info(f"Connection state: {self.pc.connectionState}")
                if self.pc.connectionState in ["failed", "closed"]:
                    logger.error(f"WebRTC connection {self.pc.connectionState}")
                    self.running = False
            
            @self.pc.on("icecandidate")
            async def on_icecandidate(event):
                if event.candidate:
                    logger.debug(f"Sending ICE candidate: {event.candidate}")
                    await websocket.send(json.dumps({
                        'type': 'ice_candidate',
                        'candidate': {
                            'candidate': event.candidate.candidate,
                            'sdpMid': event.candidate.sdpMid,
                            'sdpMLineIndex': event.candidate.sdpMLineIndex
                        }
                    }))
            
            logger.info(f"Attempting to connect to WebSocket server: {self.server_url}")
            async with websockets.connect(self.server_url, ping_interval=None) as websocket:
                logger.debug("Creating offer")
                offer = await self.create_valid_offer()
                if offer is None:
                    logger.error("Aborting connection due to invalid offer")
                    self.running = False
                    await self.pc.close()
                    return
                
                await self.pc.setLocalDescription(offer)
                logger.debug(f"Local description set: {offer.sdp[:200]}...")
                
                offer_data = {
                    'type': 'offer',
                    'sdp': self.pc.localDescription.sdp
                }
                logger.debug(f"Sending offer: {offer_data}")
                await websocket.send(json.dumps(offer_data))
                
                response = await websocket.recv()
                data = json.loads(response)
                logger.debug(f"Received response: {data}")
                
                if data.get('type') == 'answer':
                    answer = RTCSessionDescription(sdp=data['sdp'], type=data['type'])
                    await self.pc.setRemoteDescription(answer)
                    logger.info("WebRTC connection established")
                else:
                    logger.error(f"Expected answer, received: {data}")
                    self.running = False
                    await self.pc.close()
                    return
                
                async for msg in websocket:
                    try:
                        data = json.loads(msg)
                        logger.debug(f"Received WebSocket message: {data}")
                        if data.get('type') == 'ice_candidate':
                            candidate = data.get('candidate')
                            if candidate:
                                await self.pc.addIceCandidate(RTCIceCandidate(
                                    candidate=candidate['candidate'],
                                    sdpMid=candidate['sdpMid'],
                                    sdpMLineIndex=candidate['sdpMLineIndex']
                                ))
                                logger.debug("ICE candidate added")
                    except Exception as e:
                        logger.error(f"Error processing WebSocket message: {e}")
                
        except Exception as e:
            logger.error(f"WebRTC connection error: {e}\n{traceback.format_exc()}")
            self.running = False
            if self.pc:
                await self.pc.close()
    
    async def process_video_track(self, track):
        """Process incoming video frames"""
        while self.running:
            try:
                frame = await track.recv()
                img = frame.to_ndarray(format='bgr24')
                try:
                    self.frame_queue.put_nowait(img)
                except queue.Full:
                    try:
                        self.frame_queue.get_nowait()
                        self.frame_queue.put_nowait(img)
                    except queue.Empty:
                        pass
            except Exception as e:
                logger.error(f"Error processing video frame: {e}")
                await asyncio.sleep(0.1)
    
    def get_frame(self):
        """Get the latest frame (non-blocking)"""
        try:
            return self.frame_queue.get_nowait()
        except queue.Empty:
            return None
    
    def is_connected(self):
        """Check if WebRTC connection is active"""
        return self.running and self.pc and self.pc.connectionState == "connected"
    
    async def close(self):
        """Close the WebRTC connection"""
        self.running = False
        if self.pc:
            await self.pc.close()

class WebRTCVideoSource:
    """Video source that wraps WebRTC receiver to work like cv2.VideoCapture"""
    
    def __init__(self, server_url="ws://.172.20.66.142:8080/ws"):
        self.receiver = WebRTCVideoReceiver(server_url)
        self.loop = None
        self.thread = None
        self.latest_frame = None
        self.frame_count = 0
        
    def start(self):
        """Start the WebRTC receiver in a separate thread"""
        def run_receiver():
            self.loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.loop)
            self.loop.run_until_complete(self.receiver.connect_and_receive())
        
        self.thread = threading.Thread(target=run_receiver, daemon=True)
        self.thread.start()
        
        timeout = 10
        start_time = time.time()
        while not self.receiver.is_connected() and (time.time() - start_time) < timeout:
            time.sleep(0.1)
        
        if not self.receiver.is_connected():
            logger.warning("WebRTC connection not established within timeout")
        else:
            logger.info("WebRTC video source started successfully")
    
    def read(self):
        """Read a frame (mimics cv2.VideoCapture.read())"""
        frame = self.receiver.get_frame()
        if frame is not None:
            self.latest_frame = frame
            self.frame_count += 1
            return True, frame
        elif self.latest_frame is not None:
            return True, self.latest_frame.copy()
        else:
            return False, None
    
    def get(self, prop):
        """Get video properties (mimics cv2.VideoCapture.get())"""
        if prop == cv2.CAP_PROP_FRAME_WIDTH:
            return 640
        elif prop == cv2.CAP_PROP_FRAME_HEIGHT:
            return 480
        elif prop == cv2.CAP_PROP_FPS:
            return 30
        return 0
    
    def isOpened(self):
        """Check if video source is opened"""
        return self.receiver.is_connected()
    
    def release(self):
        """Release the video source"""
        if self.loop and self.receiver:
            asyncio.run_coroutine_threadsafe(self.receiver.close(), self.loop)
        if self.thread:
            self.thread.join(timeout=2)

def main():
    """Main function"""
    server_ip = "172.20.66.142"
    server_port = 8080
    server_url = f"ws://{server_ip}:{server_port}/ws"
    
    output_path = "output_webrtc.mp4"
    yolo_model_size = "nano"  # YOLOv11 model size: "nano", "small", "medium", "large", "extra"
    depth_model_size = "base"  # Depth Anything v2 model size: "small", "base", "large"
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    if device == 'cuda':
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    
    conf_threshold = 0.25
    iou_threshold = 0.45
    classes = [0,1,2,3,5,7,14,15,16,17,18,19,24,26,39,41,64,67,72,73,76]
    enable_tracking = True
    enable_bev = True
    enable_pseudo_3d = True
    process_every_n_frames = 3
    frame_count = 0
    
    logger.info(f"Using device: {device}")
    logger.info(f"Connecting to WebRTC server at: {server_url}")
    
    video_source = WebRTCVideoSource(server_url)
    out = None
    
    try:
        video_source.start()
        
        if not video_source.isOpened():
            logger.error("Failed to connect to WebRTC server")
            return
        
        logger.info("WebRTC connection established successfully")
        
        logger.info("Initializing models...")
        try:
            detector = ObjectDetector(
                model_size=yolo_model_size,
                conf_thres=conf_threshold,
                iou_thres=iou_threshold,
                classes=classes,
                device=device
            )
        except Exception as e:
            logger.error(f"Error initializing object detector: {e}")
            logger.info("Falling back to CPU for object detection")
            detector = ObjectDetector(
                model_size=yolo_model_size,
                conf_thres=conf_threshold,
                iou_thres=iou_threshold,
                classes=classes,
                device='cpu'
            )
        
        # Initialize depth estimator (default: metric, indoor)
        depth_estimator = DepthEstimator(
            model_size=depth_model_size,
            metric=True,  # Use metric depth (meters)
            scene_type='indoor'  # Change to 'outdoor' for outdoor scenes
        )
        
        bbox3d_estimator = BBox3DEstimator()
        if enable_bev:
            bev = BirdEyeView(scale=60, size=(300, 300))
        
        width = int(video_source.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video_source.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(video_source.get(cv2.CAP_PROP_FPS))
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        start_time = time.time()
        fps_display = "FPS: --"
        
        logger.info("Starting processing...")
        
        while True:
            key = cv2.waitKey(1)
            if key in [ord('q'), 27]:
                logger.info("Exiting program...")
                break
                
            try:
                ret, frame = video_source.read()
                if not ret:
                    logger.warning("No frame received from WebRTC source")
                    time.sleep(0.1)
                    continue
                
                original_frame = frame.copy()
                detection_frame = frame.copy()
                depth_frame = frame.copy()
                result_frame = frame.copy()
                
                try:
                    detection_frame, detections = detector.detect(detection_frame, track=enable_tracking)
                except Exception as e:
                    logger.error(f"Error during object detection: {e}")
                    detections = []
                    cv2.putText(detection_frame, "Detection Error", (10, 60), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                try:
                    if frame_count % process_every_n_frames == 0:
                        depth_map = depth_estimator.estimate_depth(original_frame)
                        depth_colored = depth_estimator.colorize_depth(depth_map)
                    else:
                        depth_colored = depth_colored if 'depth_colored' in locals() else np.zeros((height, width, 3), dtype=np.uint8)
                except Exception as e:
                    logger.error(f"Error during depth estimation: {e}")
                    depth_map = np.zeros((height, width), dtype=np.float32)
                    depth_colored = np.zeros((height, width, 3), dtype=np.uint8)
                    cv2.putText(depth_colored, "Depth Error", (10, 60), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                boxes_3d = []
                active_ids = []
                
                for detection in detections:
                    try:
                        bbox, score, class_id, obj_id = detection
                        class_name = detector.get_class_names()[class_id]
                        
                        if class_name.lower() in ['person', 'cat', 'dog']:
                            center_x = int((bbox[0] + bbox[2]) / 2)
                            center_y = int((bbox[1] + bbox[3]) / 2)
                            depth_value = depth_estimator.get_depth_at_point(depth_map, center_x, center_y)
                            depth_method = 'center'
                        else:
                            depth_value = depth_estimator.get_depth_in_region(depth_map, bbox, method='median')
                            depth_method = 'median'
                        
                        box_3d = {
                            'bbox_2d': bbox,
                            'depth_value': depth_value,
                            'depth_method': depth_method,
                            'class_name': class_name,
                            'object_id': obj_id,
                            'score': score
                        }
                        
                        boxes_3d.append(box_3d)
                        if obj_id is not None:
                            active_ids.append(obj_id)
                    except Exception as e:
                        logger.error(f"Error processing detection: {e}")
                        continue
                
                frame_count += 1
                if frame_count % 10 == 0:
                    end_time = time.time()
                    elapsed_time = end_time - start_time
                    fps_value = frame_count / elapsed_time
                    fps_display = f"FPS: {fps_value:.1f}"
                
                text = f"{fps_display} | Device: {device} | WebRTC"
                text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
                text_x = width - text_size[0] - 10
                cv2.putText(result_frame, text, (text_x, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                bbox3d_estimator.cleanup_trackers(active_ids)
                
                for box_3d in boxes_3d:
                    try:
                        class_name = box_3d['class_name'].lower()
                        if 'car' in class_name or 'vehicle' in class_name:
                            color = (0, 0, 255)
                        elif 'person' in class_name:
                            color = (0, 255, 0)
                        elif 'bicycle' in class_name or 'motorcycle' in class_name:
                            color = (255, 0, 0)
                        elif 'potted plant' in class_name or 'plant' in class_name:
                            color = (0, 255, 255)
                        else:
                            color = (255, 255, 255)
                        
                        result_frame = bbox3d_estimator.draw_box_3d(result_frame, box_3d, color=color)
                    except Exception as e:
                        logger.error(f"Error drawing box: {e}")
                        continue
                
                if enable_bev:
                    try:
                        bev.reset()
                        for box_3d in boxes_3d:
                            bev.draw_box(box_3d)
                        bev_image = bev.get_image()
                        
                        bev_height = height // 4
                        bev_width = bev_height
                        
                        if bev_height > 0 and bev_width > 0:
                            bev_resized = cv2.resize(bev_image, (bev_width, bev_height))
                            result_frame[height - bev_height:height, 0:bev_width] = bev_resized
                            cv2.rectangle(result_frame, 
                                         (0, height - bev_height), 
                                         (bev_width, height), 
                                         (255, 255, 255), 1)
                            cv2.putText(result_frame, "Bird's Eye View", 
                                       (10, height - bev_height + 20), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    except Exception as e:
                        logger.error(f"Error drawing BEV: {e}")
                
                out.write(result_frame)
                cv2.imshow("3D Object Detection (WebRTC)", result_frame)
                cv2.imshow("Depth Map", depth_colored)
                cv2.imshow("Object Detection", detection_frame)
                
                key = cv2.waitKey(1)
                if key in [ord('q'), 27]:
                    logger.info("Exiting program...")
                    break
            
            except Exception as e:
                logger.error(f"Error processing frame: {e}")
                key = cv2.waitKey(1)
                if key in [ord('q'), 27]:
                    logger.info("Exiting program...")
                    break
    
    finally:
        video_source.release()
        if out:
            out.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()