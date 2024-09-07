#ifndef _GAZEBO_ROS_ACTOR_POSE_H_
#define _GAZEBO_ROS_ACTOR_POSE_H_

#include <string>
#include <vector>
#include <mutex>
#include <thread>
#include <memory>

#include <rclcpp/rclcpp.hpp>
#include <rclcpp/callback_group.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>

#include <gazebo/common/Plugin.hh>
#include <gazebo/physics/physics.hh>
#include <gazebo/gazebo.hh>
// #include "gazebo/util/system.hh" // already included in gazebo.hh


namespace gazebo_plugins
{
    class GAZEBO_VISIBLE ActorPosePlugin : public gazebo::ModelPlugin
    {
        public:
            ActorPosePlugin() : ModelPlugin() {} // constructor, same as the original plugin

            ~ActorPosePlugin() // destructor
            {
                this->_update_connection.reset();
            }

            void Load(gazebo::physics::ModelPtr model, sdf::ElementPtr sdf) override;

        private:
            
            void OnUpdate();

            rclcpp::Node::SharedPtr _rosnode{nullptr};
            rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr _pub{};
            geometry_msgs::msg::PoseStamped _pose_msg{};

            std::string _topic_name;
            std::string _frame_name;
            std::string _robot_namespace{};

            gazebo::physics::ActorPtr _actor{nullptr};
            gazebo::physics::WorldPtr _world{nullptr};
            gazebo::physics::LinkPtr _link{nullptr};

            sdf::ElementPtr _sdf{nullptr};

            ignition::math::Vector3d _velocity{};

            gazebo::event::ConnectionPtr _update_connection{nullptr};
            gazebo::common::Time _last_update_time{};
            double _update_rate{0.0};

            std::mutex _mtx;
            std::string _link_name;
            const std::string _reference_frame_name{"world"};
    };
    
    GZ_REGISTER_MODEL_PLUGIN(ActorPosePlugin)
}


#endif