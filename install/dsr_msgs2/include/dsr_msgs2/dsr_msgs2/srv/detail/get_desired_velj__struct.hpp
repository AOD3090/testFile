// generated from rosidl_generator_cpp/resource/idl__struct.hpp.em
// with input from dsr_msgs2:srv/GetDesiredVelj.idl
// generated code does not contain a copyright notice

#ifndef DSR_MSGS2__SRV__DETAIL__GET_DESIRED_VELJ__STRUCT_HPP_
#define DSR_MSGS2__SRV__DETAIL__GET_DESIRED_VELJ__STRUCT_HPP_

#include <algorithm>
#include <array>
#include <memory>
#include <string>
#include <vector>

#include "rosidl_runtime_cpp/bounded_vector.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


#ifndef _WIN32
# define DEPRECATED__dsr_msgs2__srv__GetDesiredVelj_Request __attribute__((deprecated))
#else
# define DEPRECATED__dsr_msgs2__srv__GetDesiredVelj_Request __declspec(deprecated)
#endif

namespace dsr_msgs2
{

namespace srv
{

// message struct
template<class ContainerAllocator>
struct GetDesiredVelj_Request_
{
  using Type = GetDesiredVelj_Request_<ContainerAllocator>;

  explicit GetDesiredVelj_Request_(rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->structure_needs_at_least_one_member = 0;
    }
  }

  explicit GetDesiredVelj_Request_(const ContainerAllocator & _alloc, rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  {
    (void)_alloc;
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->structure_needs_at_least_one_member = 0;
    }
  }

  // field types and members
  using _structure_needs_at_least_one_member_type =
    uint8_t;
  _structure_needs_at_least_one_member_type structure_needs_at_least_one_member;


  // constant declarations

  // pointer types
  using RawPtr =
    dsr_msgs2::srv::GetDesiredVelj_Request_<ContainerAllocator> *;
  using ConstRawPtr =
    const dsr_msgs2::srv::GetDesiredVelj_Request_<ContainerAllocator> *;
  using SharedPtr =
    std::shared_ptr<dsr_msgs2::srv::GetDesiredVelj_Request_<ContainerAllocator>>;
  using ConstSharedPtr =
    std::shared_ptr<dsr_msgs2::srv::GetDesiredVelj_Request_<ContainerAllocator> const>;

  template<typename Deleter = std::default_delete<
      dsr_msgs2::srv::GetDesiredVelj_Request_<ContainerAllocator>>>
  using UniquePtrWithDeleter =
    std::unique_ptr<dsr_msgs2::srv::GetDesiredVelj_Request_<ContainerAllocator>, Deleter>;

  using UniquePtr = UniquePtrWithDeleter<>;

  template<typename Deleter = std::default_delete<
      dsr_msgs2::srv::GetDesiredVelj_Request_<ContainerAllocator>>>
  using ConstUniquePtrWithDeleter =
    std::unique_ptr<dsr_msgs2::srv::GetDesiredVelj_Request_<ContainerAllocator> const, Deleter>;
  using ConstUniquePtr = ConstUniquePtrWithDeleter<>;

  using WeakPtr =
    std::weak_ptr<dsr_msgs2::srv::GetDesiredVelj_Request_<ContainerAllocator>>;
  using ConstWeakPtr =
    std::weak_ptr<dsr_msgs2::srv::GetDesiredVelj_Request_<ContainerAllocator> const>;

  // pointer types similar to ROS 1, use SharedPtr / ConstSharedPtr instead
  // NOTE: Can't use 'using' here because GNU C++ can't parse attributes properly
  typedef DEPRECATED__dsr_msgs2__srv__GetDesiredVelj_Request
    std::shared_ptr<dsr_msgs2::srv::GetDesiredVelj_Request_<ContainerAllocator>>
    Ptr;
  typedef DEPRECATED__dsr_msgs2__srv__GetDesiredVelj_Request
    std::shared_ptr<dsr_msgs2::srv::GetDesiredVelj_Request_<ContainerAllocator> const>
    ConstPtr;

  // comparison operators
  bool operator==(const GetDesiredVelj_Request_ & other) const
  {
    if (this->structure_needs_at_least_one_member != other.structure_needs_at_least_one_member) {
      return false;
    }
    return true;
  }
  bool operator!=(const GetDesiredVelj_Request_ & other) const
  {
    return !this->operator==(other);
  }
};  // struct GetDesiredVelj_Request_

// alias to use template instance with default allocator
using GetDesiredVelj_Request =
  dsr_msgs2::srv::GetDesiredVelj_Request_<std::allocator<void>>;

// constant definitions

}  // namespace srv

}  // namespace dsr_msgs2


#ifndef _WIN32
# define DEPRECATED__dsr_msgs2__srv__GetDesiredVelj_Response __attribute__((deprecated))
#else
# define DEPRECATED__dsr_msgs2__srv__GetDesiredVelj_Response __declspec(deprecated)
#endif

namespace dsr_msgs2
{

namespace srv
{

// message struct
template<class ContainerAllocator>
struct GetDesiredVelj_Response_
{
  using Type = GetDesiredVelj_Response_<ContainerAllocator>;

  explicit GetDesiredVelj_Response_(rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      std::fill<typename std::array<double, 6>::iterator, double>(this->joint_vel.begin(), this->joint_vel.end(), 0.0);
      this->success = false;
    }
  }

  explicit GetDesiredVelj_Response_(const ContainerAllocator & _alloc, rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  : joint_vel(_alloc)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      std::fill<typename std::array<double, 6>::iterator, double>(this->joint_vel.begin(), this->joint_vel.end(), 0.0);
      this->success = false;
    }
  }

  // field types and members
  using _joint_vel_type =
    std::array<double, 6>;
  _joint_vel_type joint_vel;
  using _success_type =
    bool;
  _success_type success;

  // setters for named parameter idiom
  Type & set__joint_vel(
    const std::array<double, 6> & _arg)
  {
    this->joint_vel = _arg;
    return *this;
  }
  Type & set__success(
    const bool & _arg)
  {
    this->success = _arg;
    return *this;
  }

  // constant declarations

  // pointer types
  using RawPtr =
    dsr_msgs2::srv::GetDesiredVelj_Response_<ContainerAllocator> *;
  using ConstRawPtr =
    const dsr_msgs2::srv::GetDesiredVelj_Response_<ContainerAllocator> *;
  using SharedPtr =
    std::shared_ptr<dsr_msgs2::srv::GetDesiredVelj_Response_<ContainerAllocator>>;
  using ConstSharedPtr =
    std::shared_ptr<dsr_msgs2::srv::GetDesiredVelj_Response_<ContainerAllocator> const>;

  template<typename Deleter = std::default_delete<
      dsr_msgs2::srv::GetDesiredVelj_Response_<ContainerAllocator>>>
  using UniquePtrWithDeleter =
    std::unique_ptr<dsr_msgs2::srv::GetDesiredVelj_Response_<ContainerAllocator>, Deleter>;

  using UniquePtr = UniquePtrWithDeleter<>;

  template<typename Deleter = std::default_delete<
      dsr_msgs2::srv::GetDesiredVelj_Response_<ContainerAllocator>>>
  using ConstUniquePtrWithDeleter =
    std::unique_ptr<dsr_msgs2::srv::GetDesiredVelj_Response_<ContainerAllocator> const, Deleter>;
  using ConstUniquePtr = ConstUniquePtrWithDeleter<>;

  using WeakPtr =
    std::weak_ptr<dsr_msgs2::srv::GetDesiredVelj_Response_<ContainerAllocator>>;
  using ConstWeakPtr =
    std::weak_ptr<dsr_msgs2::srv::GetDesiredVelj_Response_<ContainerAllocator> const>;

  // pointer types similar to ROS 1, use SharedPtr / ConstSharedPtr instead
  // NOTE: Can't use 'using' here because GNU C++ can't parse attributes properly
  typedef DEPRECATED__dsr_msgs2__srv__GetDesiredVelj_Response
    std::shared_ptr<dsr_msgs2::srv::GetDesiredVelj_Response_<ContainerAllocator>>
    Ptr;
  typedef DEPRECATED__dsr_msgs2__srv__GetDesiredVelj_Response
    std::shared_ptr<dsr_msgs2::srv::GetDesiredVelj_Response_<ContainerAllocator> const>
    ConstPtr;

  // comparison operators
  bool operator==(const GetDesiredVelj_Response_ & other) const
  {
    if (this->joint_vel != other.joint_vel) {
      return false;
    }
    if (this->success != other.success) {
      return false;
    }
    return true;
  }
  bool operator!=(const GetDesiredVelj_Response_ & other) const
  {
    return !this->operator==(other);
  }
};  // struct GetDesiredVelj_Response_

// alias to use template instance with default allocator
using GetDesiredVelj_Response =
  dsr_msgs2::srv::GetDesiredVelj_Response_<std::allocator<void>>;

// constant definitions

}  // namespace srv

}  // namespace dsr_msgs2

namespace dsr_msgs2
{

namespace srv
{

struct GetDesiredVelj
{
  using Request = dsr_msgs2::srv::GetDesiredVelj_Request;
  using Response = dsr_msgs2::srv::GetDesiredVelj_Response;
};

}  // namespace srv

}  // namespace dsr_msgs2

#endif  // DSR_MSGS2__SRV__DETAIL__GET_DESIRED_VELJ__STRUCT_HPP_
