#include <iostream>

#include "gazebo_ros_actor_pose.h"


using namespace gazebo_plugins;

void ActorPosePlugin::Load(gazebo::physics::ModelPtr model, sdf::ElementPtr sdf){
    
    if (!rclcpp::ok()){
        std::cerr << "ROS is not running. Plugin will not load." << std::endl;
        return;
    }

    this->_rosnode = rclcpp::Node::make_shared("actor_pose_plugin");

    this->_sdf = sdf;
    this->_actor = boost::dynamic_pointer_cast<gazebo::physics::Actor>(model);
    this->_world = this -> _actor -> GetWorld();

    this->_link_name = std::string(this->_actor->GetName() + "_pose");
    this->_link = model->GetLink(this->_link_name);
    this->_frame_name = this -> _link_name;

    this->_last_update_time = this->_world->SimTime();

    if (!(this->_sdf->HasElement("topicName"))){
        this->_topic_name = this->_link_name;
    }
    else{
        this->_topic_name = this->_sdf->GetElement("topicName")->Get<std::string>();
    }

    if (!_sdf->HasElement("updateRate")){
        this->_update_rate = 0.0;
    }
    else{
        this->_update_rate = _sdf->GetElement("updateRate")->Get<double>();
    }

    this->_pub = this->_rosnode->create_publisher<geometry_msgs::msg::PoseStamped>(this->_topic_name, 1);
    this->_update_connection = gazebo::event::Events::ConnectWorldUpdateBegin(std::bind(&ActorPosePlugin::OnUpdate, this));

}

void ActorPosePlugin::OnUpdate(){

    if (!this->_link){
        return;
    }

    gazebo::common::Time current_time = this->_world->SimTime();

    if (current_time < this->_last_update_time){
        this->_last_update_time = current_time;
        return;
    }

    auto elapsed_time = current_time - this->_last_update_time;
    if ( (this->_update_rate > 0.0) && (elapsed_time.Double() < (1.0/this->_update_rate)) ){
        return;
    }

    std::scoped_lock lock(this->_mtx);

    ignition::math::Pose3d current_pose = this->_actor->WorldPose();

    this->_pose_msg.header.frame_id = this->_reference_frame_name;
    this->_pose_msg.header.stamp = this->_rosnode->get_clock()->now();

    this->_pose_msg.pose.position.x = current_pose.Pos().X();
    this->_pose_msg.pose.position.y = current_pose.Pos().Y();
    this->_pose_msg.pose.position.z = current_pose.Pos().Z();

    this->_pose_msg.pose.orientation.x = current_pose.Rot().X();
    this->_pose_msg.pose.orientation.y = current_pose.Rot().Y();
    this->_pose_msg.pose.orientation.z = current_pose.Rot().Z();
    this->_pose_msg.pose.orientation.w = current_pose.Rot().W();

    this->_last_update_time = current_time;

    this->_pub->publish(this->_pose_msg);

}