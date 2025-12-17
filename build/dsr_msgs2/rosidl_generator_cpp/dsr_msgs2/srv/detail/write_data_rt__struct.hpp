// generated from rosidl_generator_cpp/resource/idl__struct.hpp.em
// with input from dsr_msgs2:srv/WriteDataRt.idl
// generated code does not contain a copyright notice

#ifndef DSR_MSGS2__SRV__DETAIL__WRITE_DATA_RT__STRUCT_HPP_
#define DSR_MSGS2__SRV__DETAIL__WRITE_DATA_RT__STRUCT_HPP_

#include <algorithm>
#include <array>
#include <memory>
#include <string>
#include <vector>

#include "rosidl_runtime_cpp/bounded_vector.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


#ifndef _WIN32
# define DEPRECATED__dsr_msgs2__srv__WriteDataRt_Request __attribute__((deprecated))
#else
# define DEPRECATED__dsr_msgs2__srv__WriteDataRt_Request __declspec(deprecated)
#endif

namespace dsr_msgs2
{

namespace srv
{

// message struct
template<class ContainerAllocator>
struct WriteDataRt_Request_
{
  using Type = WriteDataRt_Request_<ContainerAllocator>;

  explicit WriteDataRt_Request_(rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      std::fill<typename std::array<double, 6>::iterator, double>(this->external_force_torque.begin(), this->external_force_torque.end(), 0.0);
      this->external_digital_input = 0l;
      this->external_digital_output = 0l;
      std::fill<typename std::array<double, 6>::iterator, double>(this->external_analog_input.begin(), this->external_analog_input.end(), 0.0);
      std::fill<typename std::array<double, 6>::iterator, double>(this->external_analog_output.begin(), this->external_analog_output.end(), 0.0);
    }
  }

  explicit WriteDataRt_Request_(const ContainerAllocator & _alloc, rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  : external_force_torque(_alloc),
    external_analog_input(_alloc),
    external_analog_output(_alloc)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      std::fill<typename std::array<double, 6>::iterator, double>(this->external_force_torque.begin(), this->external_force_torque.end(), 0.0);
      this->external_digital_input = 0l;
      this->external_digital_output = 0l;
      std::fill<typename std::array<double, 6>::iterator, double>(this->external_analog_input.begin(), this->external_analog_input.end(), 0.0);
      std::fill<typename std::array<double, 6>::iterator, double>(this->external_analog_output.begin(), this->external_analog_output.end(), 0.0);
    }
  }

  // field types and members
  using _external_force_torque_type =
    std::array<double, 6>;
  _external_force_torque_type external_force_torque;
  using _external_digital_input_type =
    int32_t;
  _external_digital_input_type external_digital_input;
  using _external_digital_output_type =
    int32_t;
  _external_digital_output_type external_digital_output;
  using _external_analog_input_type =
    std::array<double, 6>;
  _external_analog_input_type external_analog_input;
  using _external_analog_output_type =
    std::array<double, 6>;
  _external_analog_output_type external_analog_output;

  // setters for named parameter idiom
  Type & set__external_force_torque(
    const std::array<double, 6> & _arg)
  {
    this->external_force_torque = _arg;
    return *this;
  }
  Type & set__external_digital_input(
    const int32_t & _arg)
  {
    this->external_digital_input = _arg;
    return *this;
  }
  Type & set__external_digital_output(
    const int32_t & _arg)
  {
    this->external_digital_output = _arg;
    return *this;
  }
  Type & set__external_analog_input(
    const std::array<double, 6> & _arg)
  {
    this->external_analog_input = _arg;
    return *this;
  }
  Type & set__external_analog_output(
    const std::array<double, 6> & _arg)
  {
    this->external_analog_output = _arg;
    return *this;
  }

  // constant declarations

  // pointer types
  using RawPtr =
    dsr_msgs2::srv::WriteDataRt_Request_<ContainerAllocator> *;
  using ConstRawPtr =
    const dsr_msgs2::srv::WriteDataRt_Request_<ContainerAllocator> *;
  using SharedPtr =
    std::shared_ptr<dsr_msgs2::srv::WriteDataRt_Request_<ContainerAllocator>>;
  using ConstSharedPtr =
    std::shared_ptr<dsr_msgs2::srv::WriteDataRt_Request_<ContainerAllocator> const>;

  template<typename Deleter = std::default_delete<
      dsr_msgs2::srv::WriteDataRt_Request_<ContainerAllocator>>>
  using UniquePtrWithDeleter =
    std::unique_ptr<dsr_msgs2::srv::WriteDataRt_Request_<ContainerAllocator>, Deleter>;

  using UniquePtr = UniquePtrWithDeleter<>;

  template<typename Deleter = std::default_delete<
      dsr_msgs2::srv::WriteDataRt_Request_<ContainerAllocator>>>
  using ConstUniquePtrWithDeleter =
    std::unique_ptr<dsr_msgs2::srv::WriteDataRt_Request_<ContainerAllocator> const, Deleter>;
  using ConstUniquePtr = ConstUniquePtrWithDeleter<>;

  using WeakPtr =
    std::weak_ptr<dsr_msgs2::srv::WriteDataRt_Request_<ContainerAllocator>>;
  using ConstWeakPtr =
    std::weak_ptr<dsr_msgs2::srv::WriteDataRt_Request_<ContainerAllocator> const>;

  // pointer types similar to ROS 1, use SharedPtr / ConstSharedPtr instead
  // NOTE: Can't use 'using' here because GNU C++ can't parse attributes properly
  typedef DEPRECATED__dsr_msgs2__srv__WriteDataRt_Request
    std::shared_ptr<dsr_msgs2::srv::WriteDataRt_Request_<ContainerAllocator>>
    Ptr;
  typedef DEPRECATED__dsr_msgs2__srv__WriteDataRt_Request
    std::shared_ptr<dsr_msgs2::srv::WriteDataRt_Request_<ContainerAllocator> const>
    ConstPtr;

  // comparison operators
  bool operator==(const WriteDataRt_Request_ & other) const
  {
    if (this->external_force_torque != other.external_force_torque) {
      return false;
    }
    if (this->external_digital_input != other.external_digital_input) {
      return false;
    }
    if (this->external_digital_output != other.external_digital_output) {
      return false;
    }
    if (this->external_analog_input != other.external_analog_input) {
      return false;
    }
    if (this->external_analog_output != other.external_analog_output) {
      return false;
    }
    return true;
  }
  bool operator!=(const WriteDataRt_Request_ & other) const
  {
    return !this->operator==(other);
  }
};  // struct WriteDataRt_Request_

// alias to use template instance with default allocator
using WriteDataRt_Request =
  dsr_msgs2::srv::WriteDataRt_Request_<std::allocator<void>>;

// constant definitions

}  // namespace srv

}  // namespace dsr_msgs2


#ifndef _WIN32
# define DEPRECATED__dsr_msgs2__srv__WriteDataRt_Response __attribute__((deprecated))
#else
# define DEPRECATED__dsr_msgs2__srv__WriteDataRt_Response __declspec(deprecated)
#endif

namespace dsr_msgs2
{

namespace srv
{

// message struct
template<class ContainerAllocator>
struct WriteDataRt_Response_
{
  using Type = WriteDataRt_Response_<ContainerAllocator>;

  explicit WriteDataRt_Response_(rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->success = false;
    }
  }

  explicit WriteDataRt_Response_(const ContainerAllocator & _alloc, rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  {
    (void)_alloc;
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->success = false;
    }
  }

  // field types and members
  using _success_type =
    bool;
  _success_type success;

  // setters for named parameter idiom
  Type & set__success(
    const bool & _arg)
  {
    this->success = _arg;
    return *this;
  }

  // constant declarations

  // pointer types
  using RawPtr =
    dsr_msgs2::srv::WriteDataRt_Response_<ContainerAllocator> *;
  using ConstRawPtr =
    const dsr_msgs2::srv::WriteDataRt_Response_<ContainerAllocator> *;
  using SharedPtr =
    std::shared_ptr<dsr_msgs2::srv::WriteDataRt_Response_<ContainerAllocator>>;
  using ConstSharedPtr =
    std::shared_ptr<dsr_msgs2::srv::WriteDataRt_Response_<ContainerAllocator> const>;

  template<typename Deleter = std::default_delete<
      dsr_msgs2::srv::WriteDataRt_Response_<ContainerAllocator>>>
  using UniquePtrWithDeleter =
    std::unique_ptr<dsr_msgs2::srv::WriteDataRt_Response_<ContainerAllocator>, Deleter>;

  using UniquePtr = UniquePtrWithDeleter<>;

  template<typename Deleter = std::default_delete<
      dsr_msgs2::srv::WriteDataRt_Response_<ContainerAllocator>>>
  using ConstUniquePtrWithDeleter =
    std::unique_ptr<dsr_msgs2::srv::WriteDataRt_Response_<ContainerAllocator> const, Deleter>;
  using ConstUniquePtr = ConstUniquePtrWithDeleter<>;

  using WeakPtr =
    std::weak_ptr<dsr_msgs2::srv::WriteDataRt_Response_<ContainerAllocator>>;
  using ConstWeakPtr =
    std::weak_ptr<dsr_msgs2::srv::WriteDataRt_Response_<ContainerAllocator> const>;

  // pointer types similar to ROS 1, use SharedPtr / ConstSharedPtr instead
  // NOTE: Can't use 'using' here because GNU C++ can't parse attributes properly
  typedef DEPRECATED__dsr_msgs2__srv__WriteDataRt_Response
    std::shared_ptr<dsr_msgs2::srv::WriteDataRt_Response_<ContainerAllocator>>
    Ptr;
  typedef DEPRECATED__dsr_msgs2__srv__WriteDataRt_Response
    std::shared_ptr<dsr_msgs2::srv::WriteDataRt_Response_<ContainerAllocator> const>
    ConstPtr;

  // comparison operators
  bool operator==(const WriteDataRt_Response_ & other) const
  {
    if (this->success != other.success) {
      return false;
    }
    return true;
  }
  bool operator!=(const WriteDataRt_Response_ & other) const
  {
    return !this->operator==(other);
  }
};  // struct WriteDataRt_Response_

// alias to use template instance with default allocator
using WriteDataRt_Response =
  dsr_msgs2::srv::WriteDataRt_Response_<std::allocator<void>>;

// constant definitions

}  // namespace srv

}  // namespace dsr_msgs2

namespace dsr_msgs2
{

namespace srv
{

struct WriteDataRt
{
  using Request = dsr_msgs2::srv::WriteDataRt_Request;
  using Response = dsr_msgs2::srv::WriteDataRt_Response;
};

}  // namespace srv

}  // namespace dsr_msgs2

#endif  // DSR_MSGS2__SRV__DETAIL__WRITE_DATA_RT__STRUCT_HPP_
