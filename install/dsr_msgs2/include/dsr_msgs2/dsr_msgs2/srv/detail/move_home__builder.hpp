// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from dsr_msgs2:srv/MoveHome.idl
// generated code does not contain a copyright notice

#ifndef DSR_MSGS2__SRV__DETAIL__MOVE_HOME__BUILDER_HPP_
#define DSR_MSGS2__SRV__DETAIL__MOVE_HOME__BUILDER_HPP_

#include <algorithm>
#include <utility>

#include "dsr_msgs2/srv/detail/move_home__struct.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


namespace dsr_msgs2
{

namespace srv
{

namespace builder
{

class Init_MoveHome_Request_target
{
public:
  Init_MoveHome_Request_target()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  ::dsr_msgs2::srv::MoveHome_Request target(::dsr_msgs2::srv::MoveHome_Request::_target_type arg)
  {
    msg_.target = std::move(arg);
    return std::move(msg_);
  }

private:
  ::dsr_msgs2::srv::MoveHome_Request msg_;
};

}  // namespace builder

}  // namespace srv

template<typename MessageType>
auto build();

template<>
inline
auto build<::dsr_msgs2::srv::MoveHome_Request>()
{
  return dsr_msgs2::srv::builder::Init_MoveHome_Request_target();
}

}  // namespace dsr_msgs2


namespace dsr_msgs2
{

namespace srv
{

namespace builder
{

class Init_MoveHome_Response_success
{
public:
  explicit Init_MoveHome_Response_success(::dsr_msgs2::srv::MoveHome_Response & msg)
  : msg_(msg)
  {}
  ::dsr_msgs2::srv::MoveHome_Response success(::dsr_msgs2::srv::MoveHome_Response::_success_type arg)
  {
    msg_.success = std::move(arg);
    return std::move(msg_);
  }

private:
  ::dsr_msgs2::srv::MoveHome_Response msg_;
};

class Init_MoveHome_Response_res
{
public:
  Init_MoveHome_Response_res()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_MoveHome_Response_success res(::dsr_msgs2::srv::MoveHome_Response::_res_type arg)
  {
    msg_.res = std::move(arg);
    return Init_MoveHome_Response_success(msg_);
  }

private:
  ::dsr_msgs2::srv::MoveHome_Response msg_;
};

}  // namespace builder

}  // namespace srv

template<typename MessageType>
auto build();

template<>
inline
auto build<::dsr_msgs2::srv::MoveHome_Response>()
{
  return dsr_msgs2::srv::builder::Init_MoveHome_Response_res();
}

}  // namespace dsr_msgs2

#endif  // DSR_MSGS2__SRV__DETAIL__MOVE_HOME__BUILDER_HPP_
