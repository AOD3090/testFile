// generated from rosidl_generator_cpp/resource/idl__struct.hpp.em
// with input from dsr_msgs2:srv/CoordTransform.idl
// generated code does not contain a copyright notice

#ifndef DSR_MSGS2__SRV__DETAIL__COORD_TRANSFORM__STRUCT_HPP_
#define DSR_MSGS2__SRV__DETAIL__COORD_TRANSFORM__STRUCT_HPP_

#include <algorithm>
#include <array>
#include <memory>
#include <string>
#include <vector>

#include "rosidl_runtime_cpp/bounded_vector.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


#ifndef _WIN32
# define DEPRECATED__dsr_msgs2__srv__CoordTransform_Request __attribute__((deprecated))
#else
# define DEPRECATED__dsr_msgs2__srv__CoordTransform_Request __declspec(deprecated)
#endif

namespace dsr_msgs2
{

namespace srv
{

// message struct
template<class ContainerAllocator>
struct CoordTransform_Request_
{
  using Type = CoordTransform_Request_<ContainerAllocator>;

  explicit CoordTransform_Request_(rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      std::fill<typename std::array<double, 6>::iterator, double>(this->pos_in.begin(), this->pos_in.end(), 0.0);
      this->ref_in = 0;
      this->ref_out = 0;
    }
  }

  explicit CoordTransform_Request_(const ContainerAllocator & _alloc, rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  : pos_in(_alloc)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      std::fill<typename std::array<double, 6>::iterator, double>(this->pos_in.begin(), this->pos_in.end(), 0.0);
      this->ref_in = 0;
      this->ref_out = 0;
    }
  }

  // field types and members
  using _pos_in_type =
    std::array<double, 6>;
  _pos_in_type pos_in;
  using _ref_in_type =
    int8_t;
  _ref_in_type ref_in;
  using _ref_out_type =
    int8_t;
  _ref_out_type ref_out;

  // setters for named parameter idiom
  Type & set__pos_in(
    const std::array<double, 6> & _arg)
  {
    this->pos_in = _arg;
    return *this;
  }
  Type & set__ref_in(
    const int8_t & _arg)
  {
    this->ref_in = _arg;
    return *this;
  }
  Type & set__ref_out(
    const int8_t & _arg)
  {
    this->ref_out = _arg;
    return *this;
  }

  // constant declarations

  // pointer types
  using RawPtr =
    dsr_msgs2::srv::CoordTransform_Request_<ContainerAllocator> *;
  using ConstRawPtr =
    const dsr_msgs2::srv::CoordTransform_Request_<ContainerAllocator> *;
  using SharedPtr =
    std::shared_ptr<dsr_msgs2::srv::CoordTransform_Request_<ContainerAllocator>>;
  using ConstSharedPtr =
    std::shared_ptr<dsr_msgs2::srv::CoordTransform_Request_<ContainerAllocator> const>;

  template<typename Deleter = std::default_delete<
      dsr_msgs2::srv::CoordTransform_Request_<ContainerAllocator>>>
  using UniquePtrWithDeleter =
    std::unique_ptr<dsr_msgs2::srv::CoordTransform_Request_<ContainerAllocator>, Deleter>;

  using UniquePtr = UniquePtrWithDeleter<>;

  template<typename Deleter = std::default_delete<
      dsr_msgs2::srv::CoordTransform_Request_<ContainerAllocator>>>
  using ConstUniquePtrWithDeleter =
    std::unique_ptr<dsr_msgs2::srv::CoordTransform_Request_<ContainerAllocator> const, Deleter>;
  using ConstUniquePtr = ConstUniquePtrWithDeleter<>;

  using WeakPtr =
    std::weak_ptr<dsr_msgs2::srv::CoordTransform_Request_<ContainerAllocator>>;
  using ConstWeakPtr =
    std::weak_ptr<dsr_msgs2::srv::CoordTransform_Request_<ContainerAllocator> const>;

  // pointer types similar to ROS 1, use SharedPtr / ConstSharedPtr instead
  // NOTE: Can't use 'using' here because GNU C++ can't parse attributes properly
  typedef DEPRECATED__dsr_msgs2__srv__CoordTransform_Request
    std::shared_ptr<dsr_msgs2::srv::CoordTransform_Request_<ContainerAllocator>>
    Ptr;
  typedef DEPRECATED__dsr_msgs2__srv__CoordTransform_Request
    std::shared_ptr<dsr_msgs2::srv::CoordTransform_Request_<ContainerAllocator> const>
    ConstPtr;

  // comparison operators
  bool operator==(const CoordTransform_Request_ & other) const
  {
    if (this->pos_in != other.pos_in) {
      return false;
    }
    if (this->ref_in != other.ref_in) {
      return false;
    }
    if (this->ref_out != other.ref_out) {
      return false;
    }
    return true;
  }
  bool operator!=(const CoordTransform_Request_ & other) const
  {
    return !this->operator==(other);
  }
};  // struct CoordTransform_Request_

// alias to use template instance with default allocator
using CoordTransform_Request =
  dsr_msgs2::srv::CoordTransform_Request_<std::allocator<void>>;

// constant definitions

}  // namespace srv

}  // namespace dsr_msgs2


#ifndef _WIN32
# define DEPRECATED__dsr_msgs2__srv__CoordTransform_Response __attribute__((deprecated))
#else
# define DEPRECATED__dsr_msgs2__srv__CoordTransform_Response __declspec(deprecated)
#endif

namespace dsr_msgs2
{

namespace srv
{

// message struct
template<class ContainerAllocator>
struct CoordTransform_Response_
{
  using Type = CoordTransform_Response_<ContainerAllocator>;

  explicit CoordTransform_Response_(rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      std::fill<typename std::array<double, 6>::iterator, double>(this->conv_posx.begin(), this->conv_posx.end(), 0.0);
      this->success = false;
    }
  }

  explicit CoordTransform_Response_(const ContainerAllocator & _alloc, rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  : conv_posx(_alloc)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      std::fill<typename std::array<double, 6>::iterator, double>(this->conv_posx.begin(), this->conv_posx.end(), 0.0);
      this->success = false;
    }
  }

  // field types and members
  using _conv_posx_type =
    std::array<double, 6>;
  _conv_posx_type conv_posx;
  using _success_type =
    bool;
  _success_type success;

  // setters for named parameter idiom
  Type & set__conv_posx(
    const std::array<double, 6> & _arg)
  {
    this->conv_posx = _arg;
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
    dsr_msgs2::srv::CoordTransform_Response_<ContainerAllocator> *;
  using ConstRawPtr =
    const dsr_msgs2::srv::CoordTransform_Response_<ContainerAllocator> *;
  using SharedPtr =
    std::shared_ptr<dsr_msgs2::srv::CoordTransform_Response_<ContainerAllocator>>;
  using ConstSharedPtr =
    std::shared_ptr<dsr_msgs2::srv::CoordTransform_Response_<ContainerAllocator> const>;

  template<typename Deleter = std::default_delete<
      dsr_msgs2::srv::CoordTransform_Response_<ContainerAllocator>>>
  using UniquePtrWithDeleter =
    std::unique_ptr<dsr_msgs2::srv::CoordTransform_Response_<ContainerAllocator>, Deleter>;

  using UniquePtr = UniquePtrWithDeleter<>;

  template<typename Deleter = std::default_delete<
      dsr_msgs2::srv::CoordTransform_Response_<ContainerAllocator>>>
  using ConstUniquePtrWithDeleter =
    std::unique_ptr<dsr_msgs2::srv::CoordTransform_Response_<ContainerAllocator> const, Deleter>;
  using ConstUniquePtr = ConstUniquePtrWithDeleter<>;

  using WeakPtr =
    std::weak_ptr<dsr_msgs2::srv::CoordTransform_Response_<ContainerAllocator>>;
  using ConstWeakPtr =
    std::weak_ptr<dsr_msgs2::srv::CoordTransform_Response_<ContainerAllocator> const>;

  // pointer types similar to ROS 1, use SharedPtr / ConstSharedPtr instead
  // NOTE: Can't use 'using' here because GNU C++ can't parse attributes properly
  typedef DEPRECATED__dsr_msgs2__srv__CoordTransform_Response
    std::shared_ptr<dsr_msgs2::srv::CoordTransform_Response_<ContainerAllocator>>
    Ptr;
  typedef DEPRECATED__dsr_msgs2__srv__CoordTransform_Response
    std::shared_ptr<dsr_msgs2::srv::CoordTransform_Response_<ContainerAllocator> const>
    ConstPtr;

  // comparison operators
  bool operator==(const CoordTransform_Response_ & other) const
  {
    if (this->conv_posx != other.conv_posx) {
      return false;
    }
    if (this->success != other.success) {
      return false;
    }
    return true;
  }
  bool operator!=(const CoordTransform_Response_ & other) const
  {
    return !this->operator==(other);
  }
};  // struct CoordTransform_Response_

// alias to use template instance with default allocator
using CoordTransform_Response =
  dsr_msgs2::srv::CoordTransform_Response_<std::allocator<void>>;

// constant definitions

}  // namespace srv

}  // namespace dsr_msgs2

namespace dsr_msgs2
{

namespace srv
{

struct CoordTransform
{
  using Request = dsr_msgs2::srv::CoordTransform_Request;
  using Response = dsr_msgs2::srv::CoordTransform_Response;
};

}  // namespace srv

}  // namespace dsr_msgs2

#endif  // DSR_MSGS2__SRV__DETAIL__COORD_TRANSFORM__STRUCT_HPP_
