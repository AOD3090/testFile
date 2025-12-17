// generated from rosidl_generator_cpp/resource/idl__struct.hpp.em
// with input from dsr_msgs2:srv/MoveLine.idl
// generated code does not contain a copyright notice

#ifndef DSR_MSGS2__SRV__DETAIL__MOVE_LINE__STRUCT_HPP_
#define DSR_MSGS2__SRV__DETAIL__MOVE_LINE__STRUCT_HPP_

#include <algorithm>
#include <array>
#include <memory>
#include <string>
#include <vector>

#include "rosidl_runtime_cpp/bounded_vector.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


#ifndef _WIN32
# define DEPRECATED__dsr_msgs2__srv__MoveLine_Request __attribute__((deprecated))
#else
# define DEPRECATED__dsr_msgs2__srv__MoveLine_Request __declspec(deprecated)
#endif

namespace dsr_msgs2
{

namespace srv
{

// message struct
template<class ContainerAllocator>
struct MoveLine_Request_
{
  using Type = MoveLine_Request_<ContainerAllocator>;

  explicit MoveLine_Request_(rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      std::fill<typename std::array<double, 6>::iterator, double>(this->pos.begin(), this->pos.end(), 0.0);
      std::fill<typename std::array<double, 2>::iterator, double>(this->vel.begin(), this->vel.end(), 0.0);
      std::fill<typename std::array<double, 2>::iterator, double>(this->acc.begin(), this->acc.end(), 0.0);
      this->time = 0.0;
      this->radius = 0.0;
      this->ref = 0;
      this->mode = 0;
      this->blend_type = 0;
      this->sync_type = 0;
    }
  }

  explicit MoveLine_Request_(const ContainerAllocator & _alloc, rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  : pos(_alloc),
    vel(_alloc),
    acc(_alloc)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      std::fill<typename std::array<double, 6>::iterator, double>(this->pos.begin(), this->pos.end(), 0.0);
      std::fill<typename std::array<double, 2>::iterator, double>(this->vel.begin(), this->vel.end(), 0.0);
      std::fill<typename std::array<double, 2>::iterator, double>(this->acc.begin(), this->acc.end(), 0.0);
      this->time = 0.0;
      this->radius = 0.0;
      this->ref = 0;
      this->mode = 0;
      this->blend_type = 0;
      this->sync_type = 0;
    }
  }

  // field types and members
  using _pos_type =
    std::array<double, 6>;
  _pos_type pos;
  using _vel_type =
    std::array<double, 2>;
  _vel_type vel;
  using _acc_type =
    std::array<double, 2>;
  _acc_type acc;
  using _time_type =
    double;
  _time_type time;
  using _radius_type =
    double;
  _radius_type radius;
  using _ref_type =
    int8_t;
  _ref_type ref;
  using _mode_type =
    int8_t;
  _mode_type mode;
  using _blend_type_type =
    int8_t;
  _blend_type_type blend_type;
  using _sync_type_type =
    int8_t;
  _sync_type_type sync_type;

  // setters for named parameter idiom
  Type & set__pos(
    const std::array<double, 6> & _arg)
  {
    this->pos = _arg;
    return *this;
  }
  Type & set__vel(
    const std::array<double, 2> & _arg)
  {
    this->vel = _arg;
    return *this;
  }
  Type & set__acc(
    const std::array<double, 2> & _arg)
  {
    this->acc = _arg;
    return *this;
  }
  Type & set__time(
    const double & _arg)
  {
    this->time = _arg;
    return *this;
  }
  Type & set__radius(
    const double & _arg)
  {
    this->radius = _arg;
    return *this;
  }
  Type & set__ref(
    const int8_t & _arg)
  {
    this->ref = _arg;
    return *this;
  }
  Type & set__mode(
    const int8_t & _arg)
  {
    this->mode = _arg;
    return *this;
  }
  Type & set__blend_type(
    const int8_t & _arg)
  {
    this->blend_type = _arg;
    return *this;
  }
  Type & set__sync_type(
    const int8_t & _arg)
  {
    this->sync_type = _arg;
    return *this;
  }

  // constant declarations

  // pointer types
  using RawPtr =
    dsr_msgs2::srv::MoveLine_Request_<ContainerAllocator> *;
  using ConstRawPtr =
    const dsr_msgs2::srv::MoveLine_Request_<ContainerAllocator> *;
  using SharedPtr =
    std::shared_ptr<dsr_msgs2::srv::MoveLine_Request_<ContainerAllocator>>;
  using ConstSharedPtr =
    std::shared_ptr<dsr_msgs2::srv::MoveLine_Request_<ContainerAllocator> const>;

  template<typename Deleter = std::default_delete<
      dsr_msgs2::srv::MoveLine_Request_<ContainerAllocator>>>
  using UniquePtrWithDeleter =
    std::unique_ptr<dsr_msgs2::srv::MoveLine_Request_<ContainerAllocator>, Deleter>;

  using UniquePtr = UniquePtrWithDeleter<>;

  template<typename Deleter = std::default_delete<
      dsr_msgs2::srv::MoveLine_Request_<ContainerAllocator>>>
  using ConstUniquePtrWithDeleter =
    std::unique_ptr<dsr_msgs2::srv::MoveLine_Request_<ContainerAllocator> const, Deleter>;
  using ConstUniquePtr = ConstUniquePtrWithDeleter<>;

  using WeakPtr =
    std::weak_ptr<dsr_msgs2::srv::MoveLine_Request_<ContainerAllocator>>;
  using ConstWeakPtr =
    std::weak_ptr<dsr_msgs2::srv::MoveLine_Request_<ContainerAllocator> const>;

  // pointer types similar to ROS 1, use SharedPtr / ConstSharedPtr instead
  // NOTE: Can't use 'using' here because GNU C++ can't parse attributes properly
  typedef DEPRECATED__dsr_msgs2__srv__MoveLine_Request
    std::shared_ptr<dsr_msgs2::srv::MoveLine_Request_<ContainerAllocator>>
    Ptr;
  typedef DEPRECATED__dsr_msgs2__srv__MoveLine_Request
    std::shared_ptr<dsr_msgs2::srv::MoveLine_Request_<ContainerAllocator> const>
    ConstPtr;

  // comparison operators
  bool operator==(const MoveLine_Request_ & other) const
  {
    if (this->pos != other.pos) {
      return false;
    }
    if (this->vel != other.vel) {
      return false;
    }
    if (this->acc != other.acc) {
      return false;
    }
    if (this->time != other.time) {
      return false;
    }
    if (this->radius != other.radius) {
      return false;
    }
    if (this->ref != other.ref) {
      return false;
    }
    if (this->mode != other.mode) {
      return false;
    }
    if (this->blend_type != other.blend_type) {
      return false;
    }
    if (this->sync_type != other.sync_type) {
      return false;
    }
    return true;
  }
  bool operator!=(const MoveLine_Request_ & other) const
  {
    return !this->operator==(other);
  }
};  // struct MoveLine_Request_

// alias to use template instance with default allocator
using MoveLine_Request =
  dsr_msgs2::srv::MoveLine_Request_<std::allocator<void>>;

// constant definitions

}  // namespace srv

}  // namespace dsr_msgs2


#ifndef _WIN32
# define DEPRECATED__dsr_msgs2__srv__MoveLine_Response __attribute__((deprecated))
#else
# define DEPRECATED__dsr_msgs2__srv__MoveLine_Response __declspec(deprecated)
#endif

namespace dsr_msgs2
{

namespace srv
{

// message struct
template<class ContainerAllocator>
struct MoveLine_Response_
{
  using Type = MoveLine_Response_<ContainerAllocator>;

  explicit MoveLine_Response_(rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->success = false;
    }
  }

  explicit MoveLine_Response_(const ContainerAllocator & _alloc, rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
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
    dsr_msgs2::srv::MoveLine_Response_<ContainerAllocator> *;
  using ConstRawPtr =
    const dsr_msgs2::srv::MoveLine_Response_<ContainerAllocator> *;
  using SharedPtr =
    std::shared_ptr<dsr_msgs2::srv::MoveLine_Response_<ContainerAllocator>>;
  using ConstSharedPtr =
    std::shared_ptr<dsr_msgs2::srv::MoveLine_Response_<ContainerAllocator> const>;

  template<typename Deleter = std::default_delete<
      dsr_msgs2::srv::MoveLine_Response_<ContainerAllocator>>>
  using UniquePtrWithDeleter =
    std::unique_ptr<dsr_msgs2::srv::MoveLine_Response_<ContainerAllocator>, Deleter>;

  using UniquePtr = UniquePtrWithDeleter<>;

  template<typename Deleter = std::default_delete<
      dsr_msgs2::srv::MoveLine_Response_<ContainerAllocator>>>
  using ConstUniquePtrWithDeleter =
    std::unique_ptr<dsr_msgs2::srv::MoveLine_Response_<ContainerAllocator> const, Deleter>;
  using ConstUniquePtr = ConstUniquePtrWithDeleter<>;

  using WeakPtr =
    std::weak_ptr<dsr_msgs2::srv::MoveLine_Response_<ContainerAllocator>>;
  using ConstWeakPtr =
    std::weak_ptr<dsr_msgs2::srv::MoveLine_Response_<ContainerAllocator> const>;

  // pointer types similar to ROS 1, use SharedPtr / ConstSharedPtr instead
  // NOTE: Can't use 'using' here because GNU C++ can't parse attributes properly
  typedef DEPRECATED__dsr_msgs2__srv__MoveLine_Response
    std::shared_ptr<dsr_msgs2::srv::MoveLine_Response_<ContainerAllocator>>
    Ptr;
  typedef DEPRECATED__dsr_msgs2__srv__MoveLine_Response
    std::shared_ptr<dsr_msgs2::srv::MoveLine_Response_<ContainerAllocator> const>
    ConstPtr;

  // comparison operators
  bool operator==(const MoveLine_Response_ & other) const
  {
    if (this->success != other.success) {
      return false;
    }
    return true;
  }
  bool operator!=(const MoveLine_Response_ & other) const
  {
    return !this->operator==(other);
  }
};  // struct MoveLine_Response_

// alias to use template instance with default allocator
using MoveLine_Response =
  dsr_msgs2::srv::MoveLine_Response_<std::allocator<void>>;

// constant definitions

}  // namespace srv

}  // namespace dsr_msgs2

namespace dsr_msgs2
{

namespace srv
{

struct MoveLine
{
  using Request = dsr_msgs2::srv::MoveLine_Request;
  using Response = dsr_msgs2::srv::MoveLine_Response;
};

}  // namespace srv

}  // namespace dsr_msgs2

#endif  // DSR_MSGS2__SRV__DETAIL__MOVE_LINE__STRUCT_HPP_
