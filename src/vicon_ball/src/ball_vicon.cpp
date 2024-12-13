#include <cstdio>
#include <string>
#include <iostream>
#include <memory>
// ROS includes
#include "rclcpp/rclcpp.hpp"
#include "geometry_msgs/msg/point.hpp"
#include "visualization_msgs/msg/marker.hpp"

// Vicon includes
#include "DataStreamClient.h"

using namespace ViconDataStreamSDK::CPP;

class BallCenterNode : public rclcpp::Node
{
  public:

  BallCenterNode() 
  :Node("vicon")
  {
    // get parameters
    this->declare_parameter<std::string>("hostname", "10.1.108.137");
    this->declare_parameter<int>("buffer_size", 200);
    this->declare_parameter<std::string>("namespace", "vicon");

    this->get_parameter("hostname", hostname);
    this->get_parameter("buffer_size", buffer_size);
    this->get_parameter("namespace", ns_name);

    subject_name = "Kutija";

    // ROS
    publisher = this->create_publisher<geometry_msgs::msg::Point>("/vicon/kutija", rclcpp::QoS(10));
    marker_pub = this->create_publisher<visualization_msgs::msg::Marker>("visualization_marker",1);  

    marker_msg.header.frame_id = "map";
    marker_msg.header.stamp = this->get_clock().get()->now();
    marker_msg.ns = "basic_shapes";
    marker_msg.id = 0;

    marker_msg.type = visualization_msgs::msg::Marker::SPHERE;
    marker_msg.action = visualization_msgs::msg::Marker::ADD;

    marker_msg.pose.position.x = 0; 
    marker_msg.pose.position.y = 0; 
    marker_msg.pose.position.z = 0;

    marker_msg.pose.orientation.x = 0; 
    marker_msg.pose.orientation.y = 0; 
    marker_msg.pose.orientation.z = 0; 
    marker_msg.pose.orientation.w = 1;

    marker_msg.scale.x = 0.2; 
    marker_msg.scale.y = 0.2; 
    marker_msg.scale.z = 0.2;

    marker_msg.color.r = 0.0; 
    marker_msg.color.g = 1.0; 
    marker_msg.color.b = 0.0; 
    marker_msg.color.a = 1.0;
    marker_msg.lifetime = rclcpp::Duration::from_nanoseconds(0);

  }

  // Initialises the connection to the DataStream server
  bool connect() 
  {
     // connect to server
    std::string msg = "Connecting to " + hostname + " ...";
    std::cout << msg << std::endl;
    int counter = 0;

    while (!vicon_client.IsConnected().Connected)
    {
        bool ok = (vicon_client.Connect(hostname).Result == Result::Success);
        if (!ok)
        {
            counter++;
            msg = "Connect failed, reconnecting (" + std::to_string(counter) + ")...";
            std::cout << msg << std::endl;
            sleep(1);
        }
    }
    msg = "Connection successfully established with " + hostname;
    std::cout << msg << std::endl;

    // perform further initialization
    vicon_client.EnableSegmentData();
    vicon_client.EnableMarkerData();
    vicon_client.EnableUnlabeledMarkerData();
    vicon_client.EnableMarkerRayData();
    vicon_client.EnableDeviceData();
    vicon_client.EnableDebugData();

    vicon_client.SetStreamMode(StreamMode::ClientPull);
    vicon_client.SetBufferSize(buffer_size);

    msg = "Initialization complete";
    std::cout << msg << std::endl;

    return true;
  }

  // Stops the current connection to a DataStream server (if any).
  bool disconnect() 
  {
    if (!vicon_client.IsConnected().Connected)
        return true;
    sleep(1);
    vicon_client.DisableSegmentData();
    vicon_client.DisableMarkerData();
    vicon_client.DisableUnlabeledMarkerData();
    vicon_client.DisableDeviceData();
    vicon_client.DisableCentroidData();
    std::string msg = "Disconnecting from " + hostname + "...";
    std::cout << msg << std::endl;
    vicon_client.Disconnect();
    msg = "Successfully disconnected";
    std::cout << msg << std::endl;

    if (!vicon_client.IsConnected().Connected)
        return true;
    return false;
  }

  void initialize()
  {
     // --------------------------
    vicon_client.GetFrame();
    marker_count = vicon_client.GetMarkerCount(subject_name).MarkerCount;
    std::cout << "Got number of markers: " << marker_count << std::endl;

    for (unsigned int i = 0; i < marker_count; i++)
    {
      std::string marker_name = vicon_client.GetMarkerName(subject_name, i).MarkerName;
      
      marker_names.push_back(marker_name);
    }
    // -------------------
  }
  void publish()
  {
    vicon_client.GetFrame();
    // Output_GetFrameNumber frame_number = vicon_client.GetFrameNumber();
  
    // msg.x = 0;
    // msg.y = 0;
    // msg.z = 0;

    for (auto name : marker_names)
    {
      Output_GetMarkerGlobalTranslation marker_pos = vicon_client.GetMarkerGlobalTranslation(subject_name, name);

      msg.x += marker_pos.Translation[0];
      msg.y += marker_pos.Translation[1];
      msg.z += marker_pos.Translation[2];
    }

    msg.x = msg.x / this->marker_count;
    msg.y = msg.y / this->marker_count;
    msg.z = msg.z / this->marker_count;

    publisher->publish(msg);

    // Update timestamp
    marker_msg.action = visualization_msgs::msg::Marker::MODIFY;
    marker_msg.header.stamp = this->get_clock().get()->now();

    // Update pose
    marker_msg.pose.position.x = msg.x / 1000.0; 
    marker_msg.pose.position.y = msg.y / 1000.0; 
    marker_msg.pose.position.z = msg.z / 1000.0;

    marker_msg.pose.orientation.x = 0; 
    marker_msg.pose.orientation.y = 0; 
    marker_msg.pose.orientation.z = 0; 
    marker_msg.pose.orientation.w = 1;

    marker_pub->publish(marker_msg);
  }


private:

    ViconDataStreamSDK::CPP::Client vicon_client;
    std::string hostname;
    unsigned int buffer_size;
    std::string ns_name;

    std::string subject_name;
    unsigned int marker_count;
    std::vector<std::string> marker_names;

    // ---- ROS ----
    geometry_msgs::msg::Point msg;
    rclcpp::Publisher<geometry_msgs::msg::Point>::SharedPtr publisher;

    visualization_msgs::msg::Marker marker_msg;
    rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr marker_pub;

};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);

  auto node = std::make_shared<BallCenterNode>();
  node->connect();
  node->initialize();

  while (rclcpp::ok())
  {
    node->publish();
  }

  node->disconnect();
  rclcpp::shutdown();

  return 0;
}
