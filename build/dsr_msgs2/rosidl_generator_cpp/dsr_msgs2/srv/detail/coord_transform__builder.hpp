// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from dsr_msgs2:srv/CoordTransform.idl
// generated code does not contain a copyright notice

#ifndef DSR_MSGS2__SRV__DETAIL__COORD_TRANSFORM__BUILDER_HPP_
#define DSR_MSGS2__SRV__DETAIL__COORD_TRANSFORM__BUILDER_HPP_

#include <algorithm>
#include <utility>

#include "dsr_msgs2/srv/detail/coord_transform__struct.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


namespace dsr_msgs2
{

namespace srv
{

namespace builder
{

class Init_CoordTransform_Request_ref_out
{
public:
  explicit Init_CoordTransform_Request_ref_out(::dsr_msgs2::srv::CoordTransform_Request & msg)
  : msg_(msg)
  {}
  ::dsr_msgs2::srv::CoordTransform_Request ref_out(::dsr_msgs2::srv::CoordTransform_Request::_ref_out_type arg)
  {
    msg_.ref_out = std::move(arg);
    return std::move(msg_);
  }

private:
  ::dsr_msgs2::srv::CoordTransform_Request msg_;
};

class Init_CoordTransform_Request_ref_in
{
public:
  explicit Init_CoordTransform_Request_ref_in(::dsr_msgs2::srv::CoordTransform_Request & msg)
  : msg_(msg)
  {}
  Init_CoordTransform_Request_ref_out ref_in(::dsr_msgs2::srv::CoordTransform_Request::_ref_in_type arg)
  {
    msg_.ref_in = std::move(arg);
    return Init_CoordTransform_Request_ref_out(msg_);
  }

private:
  ::dsr_msgs2::srv::CoordTransform_Request msg_;
};

class Init_CoordTransform_Request_pos_in
{
public:
  Init_CoordTransform_Request_pos_in()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_CoordTransform_Request_ref_in pos_in(::dsr_msgs2::srv::CoordTransform_Request::_pos_in_type arg)
  {
    msg_.pos_in = std::move(arg);
    return Init_CoordTransform_Request_ref_in(msg_);
  }

private:
  ::dsr_msgs2::srv::CoordTransform_Request msg_;
};

}  // namespace builder

}  // namespace srv

template<typename MessageType>
auto build();

template<>
inline
auto build<::dsr_msgs2::srv::CoordTransform_Request>()
{
  return dsr_msgs2::srv::builder::Init_CoordTransform_Request_pos_in();
}

}  // namespace dsr_msgs2


namespace dsr_msgs2
{

namespace srv
{

namespace builder
{

class Init_CoordTransform_Response_success
{
public:
  explicit Init_CoordTransform_Response_success(::dsr_msgs2::srv::CoordTransform_Response & msg)
  : msg_(msg)
  {}
  ::dsr_msgs2::srv::CoordTransform_Response success(::dsr_msgs2::srv::CoordTransform_Response::_success_type arg)
  {
    msg_.success = std::move(arg);
    return std::move(msg_);
  }

private:
  ::dsr_msgs2::srv::CoordTransform_Response msg_;
};

class Init_CoordTransform_Response_conv_posx
{
public:
  Init_CoordTransform_Response_conv_posx()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_CoordTransform_Response_success conv_posx(::dsr_msgs2::srv::CoordTransform_Response::_conv_posx_type arg)
  {
    msg_.conv_posx = std::move(arg);
    return Init_CoordTransform_Response_success(msg_);
  }

private:
  ::dsr_msgs2::srv::CoordTransform_Response msg_;
};

}  // namespace builder

}  // namespace srv

template<typename MessageType>
auto build();

template<>
inline
auto build<::dsr_msgs2::srv::CoordTransform_Response>()
{
  return dsr_msgs2::srv::builder::Init_CoordTransform_Response_conv_posx();
}

}  // namespace dsr_msgs2

#endif  // DSR_MSGS2__SRV__DETAIL__COORD_TRANSFORM__BUILDER_HPP_
